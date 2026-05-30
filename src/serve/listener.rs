//! Unix domain socket listener for the `nark serve` daemon.
//!
//! Slice 2.2: bind a `UnixListener`, accept connections, and answer a
//! line-oriented `ping` with `pong`. The socket file is created under a
//! `0700` parent directory and removed on shutdown (ctrl-c or listener drop).
//!
//! Slice 2.5 assembles the authenticated path: each connection's peer uid is
//! extracted ([`super::peercred::peer_uid`]), resolved to an agent via the
//! injected [`AgentMap`], and only known agents are served `ping`->`pong`.
//! Unknown / forged uids get an `unauthorized` line and the connection is
//! closed (fail closed).
//!
//! Before binding, a pre-bind lstat guard ([`guard_preexisting_socket_path`])
//! refuses to serve if a symlink or a foreign-owned file already sits at the
//! socket path; only a self-owned stale socket is then unlinked and rebound.
//! Full liveness probing lands in a later slice.
//!
//! Each per-connection request read is bounded by [`READ_TIMEOUT`] so an idle
//! peer that never sends a line cannot park its task forever.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use super::AgentMap;
use super::authz::guard_preexisting_socket_path;
use super::peercred::peer_uid;

/// How long to wait for an authenticated peer to send its request line before
/// the connection is logged and closed. A peer that connects and never sends a
/// newline would otherwise park the spawned task forever.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Resolve the socket path for the daemon.
///
/// When `socket` is provided it is used verbatim; otherwise the path defaults
/// to `<vault_dir>/run/nark.sock`. The default deliberately nests the socket in
/// a dedicated `run/` subdir: [`ensure_socket_dir`] chmods the socket's *parent*
/// to `0700`, and that parent must never be the vault root (`~/.ark`), which
/// holds `objects/`, `registry.db`, and `config.toml` shared by the whole CLI.
pub fn resolve_socket_path(vault_dir: &Path, socket: Option<String>) -> PathBuf {
    match socket {
        Some(s) => PathBuf::from(s),
        None => vault_dir.join("run").join("nark.sock"),
    }
}

/// Ensure the parent directory of `socket_path` exists at mode `0700`.
///
/// This chmods the socket's *immediate parent* (the dedicated `run/` dir for the
/// default path). It must only ever be pointed at a socket-only directory — the
/// default path keeps the socket out of the vault root so this never alters the
/// vault root's mode. With a `--socket` override the operator owns that choice.
fn ensure_socket_dir(socket_path: &Path) -> Result<()> {
    let parent = socket_path
        .parent()
        .context("socket path has no parent directory")?;
    if !parent.exists() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating socket directory {}", parent.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o700);
        std::fs::set_permissions(parent, perms)
            .with_context(|| format!("setting 0700 on {}", parent.display()))?;
    }
    Ok(())
}

/// Remove a pre-existing socket file (stale-cleanup).
fn unlink_existing(socket_path: &Path) -> Result<()> {
    match std::fs::remove_file(socket_path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => {
            Err(e).with_context(|| format!("removing stale socket {}", socket_path.display()))
        }
    }
}

/// A bound listener whose socket file is removed when dropped.
///
/// Holding the `UnixListener` and the bound path together lets us guarantee the
/// socket file is cleaned up on both graceful shutdown and unwinding.
pub struct BoundListener {
    listener: UnixListener,
    path: PathBuf,
}

impl BoundListener {
    /// Bind a fresh listener at `socket_path`.
    ///
    /// Steps, in order:
    /// 1. create the `0700` parent directory ([`ensure_socket_dir`]) — primary
    ///    containment;
    /// 2. **pre-bind** lstat guard ([`guard_preexisting_socket_path`]): refuse if
    ///    a symlink or a foreign-owned file already sits at the path. This must
    ///    run before unlink+bind, because once we bind our own socket the path is
    ///    owned by us and the check can no longer fire;
    /// 3. unlink any (now-verified self-owned) stale socket;
    /// 4. bind the `UnixListener`.
    pub fn bind(socket_path: &Path) -> Result<Self> {
        ensure_socket_dir(socket_path)?;
        guard_preexisting_socket_path(socket_path)?;
        unlink_existing(socket_path)?;
        let listener = UnixListener::bind(socket_path)
            .with_context(|| format!("binding unix socket {}", socket_path.display()))?;
        Ok(Self {
            listener,
            path: socket_path.to_path_buf(),
        })
    }

    /// The path the listener is bound to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Accept-loop until `shutdown` resolves. Each connection is handled by
    /// reading a single line and replying `pong\n` to `ping`.
    ///
    /// This is the unauthenticated baseline from slice 2.2; the daemon path now
    /// uses [`Self::serve_authenticated_until`]. It is retained (and exercised
    /// by tests) as the documented pre-auth ping/pong primitive.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn serve_until<F>(&self, shutdown: F) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                accepted = self.listener.accept() => {
                    let (stream, _addr) = accepted.context("accepting connection")?;
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream).await {
                            eprintln!("nark serve: connection error: {e:#}");
                        }
                    });
                }
                () = &mut shutdown => {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Authenticated accept-loop until `shutdown` resolves.
    ///
    /// For each connection the peer's uid is extracted and resolved against
    /// `agents`. Known uids are served `ping`->`pong`; unknown / forged uids
    /// receive an `unauthorized` line and the connection is closed (fail
    /// closed). This is the assembled Phase 2 serve path.
    ///
    /// The per-connection request read is bounded by [`READ_TIMEOUT`]; a peer
    /// that connects and never sends a line is closed rather than parking its
    /// task forever.
    pub async fn serve_authenticated_until<F>(&self, agents: AgentMap, shutdown: F) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        self.serve_authenticated_until_with_timeout(agents, READ_TIMEOUT, shutdown)
            .await
    }

    /// As [`Self::serve_authenticated_until`], but with an explicit per-connection
    /// read timeout. Lets tests drive the timeout path with a short duration
    /// without waiting the production [`READ_TIMEOUT`].
    pub async fn serve_authenticated_until_with_timeout<F>(
        &self,
        agents: AgentMap,
        read_timeout: Duration,
        shutdown: F,
    ) -> Result<()>
    where
        F: std::future::Future<Output = ()>,
    {
        let agents = Arc::new(agents);
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                accepted = self.listener.accept() => {
                    let (stream, _addr) = accepted.context("accepting connection")?;
                    let agents = Arc::clone(&agents);
                    tokio::spawn(async move {
                        if let Err(e) =
                            handle_authenticated_connection(stream, &agents, read_timeout).await
                        {
                            eprintln!("nark serve: connection error: {e:#}");
                        }
                    });
                }
                () = &mut shutdown => {
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for BoundListener {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Handle a single connection: read one line, answer `ping` with `pong`.
///
/// Unauthenticated baseline from slice 2.2, superseded on the daemon path by
/// [`handle_authenticated_connection`]; retained as a tested primitive.
#[cfg_attr(not(test), allow(dead_code))]
async fn handle_connection(stream: UnixStream) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let n = reader
        .read_line(&mut line)
        .await
        .context("reading request line")?;
    if n == 0 {
        return Ok(());
    }
    if line.trim_end() == "ping" {
        write_half
            .write_all(b"pong\n")
            .await
            .context("writing pong")?;
        write_half.flush().await.context("flushing pong")?;
    }
    Ok(())
}

/// Handle a single connection with peer authentication.
///
/// Extracts the peer uid, resolves it to an agent via `agents`, and only then
/// serves `ping`->`pong`. Unknown uids get an `unauthorized` line and the
/// connection is closed without serving anything (fail closed).
///
/// The request read is bounded by `read_timeout`: a peer that authenticates but
/// never sends a request line is logged and the connection closed, instead of
/// parking the spawned task forever. The accept loop is unaffected because this
/// runs inside the per-connection task.
async fn handle_authenticated_connection(
    stream: UnixStream,
    agents: &AgentMap,
    read_timeout: Duration,
) -> Result<()> {
    let uid = peer_uid(&stream).context("extracting peer uid")?;
    let agent = match agents.resolve(uid) {
        Some(agent) => agent.to_string(),
        None => {
            eprintln!("nark serve: rejecting unauthorized peer (uid {uid})");
            let (_read_half, mut write_half) = stream.into_split();
            write_half
                .write_all(b"unauthorized\n")
                .await
                .context("writing rejection")?;
            write_half.flush().await.context("flushing rejection")?;
            // Dropping `write_half` (and the moved read half) closes the
            // connection so the client sees EOF after the rejection.
            return Ok(());
        }
    };

    eprintln!("nark serve: authenticated {agent} (uid {uid})");

    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let n = match tokio::time::timeout(read_timeout, reader.read_line(&mut line)).await {
        Ok(read) => read.context("reading request line")?,
        Err(_elapsed) => {
            eprintln!(
                "nark serve: {agent} (uid {uid}) sent no request within {}s; closing",
                read_timeout.as_secs()
            );
            // Dropping the halves closes the connection.
            return Ok(());
        }
    };
    if n == 0 {
        return Ok(());
    }
    if line.trim_end() == "ping" {
        write_half
            .write_all(b"pong\n")
            .await
            .context("writing pong")?;
        write_half.flush().await.context("flushing pong")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::oneshot;

    fn temp_socket_dir() -> PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "nark-serve-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        );
        base.join(unique)
    }

    #[test]
    fn resolve_defaults_to_vault_run_nark_sock() {
        let vault = Path::new("/tmp/some-vault");
        let resolved = resolve_socket_path(vault, None);
        assert_eq!(resolved, Path::new("/tmp/some-vault/run/nark.sock"));
    }

    #[test]
    fn resolve_uses_override() {
        let vault = Path::new("/tmp/some-vault");
        let resolved = resolve_socket_path(vault, Some("/run/custom.sock".to_string()));
        assert_eq!(resolved, Path::new("/run/custom.sock"));
    }

    /// End-to-end Phase 2 exit: an authenticated `ping` round-trips to `pong`
    /// when the connecting uid is a known agent, and an unknown uid is rejected
    /// at the application layer with an `unauthorized` line.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_ping_known_uid_gets_pong() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Register the *test process's own* uid as a known agent so the
        // connecting client (this process) authenticates as "tester".
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        stream.write_all(b"ping\n").await.expect("write ping");
        stream.flush().await.expect("flush ping");

        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).await.expect("read pong");
        assert_eq!(&buf, b"pong\n", "known uid should get pong");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn unknown_uid_is_rejected_with_unauthorized() {
        use super::super::AgentMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Empty map: the connecting uid is unknown -> rejected at app layer.
        let agents = AgentMap::default();

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until(agents, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        // Send a ping; the server must reject before answering.
        stream.write_all(b"ping\n").await.expect("write ping");
        stream.flush().await.expect("flush ping");

        // Read whatever the server sends back: it must be the rejection line,
        // and there must be no `pong`.
        let mut reply = String::new();
        let mut reader = BufReader::new(stream);
        reader
            .read_line(&mut reply)
            .await
            .expect("read rejection line");
        assert_eq!(
            reply, "unauthorized\n",
            "unknown uid should be rejected at the application layer"
        );

        // Connection should be closed right after the rejection (EOF).
        let mut rest = String::new();
        let n = reader
            .read_line(&mut rest)
            .await
            .expect("read after reject");
        assert_eq!(n, 0, "server should close the connection after rejecting");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn ping_pong_over_uds_and_socket_lifecycle() {
        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");

        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Socket directory must be mode 0700.
        let dir_mode = std::fs::metadata(&dir)
            .expect("stat socket dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700, "socket dir should be 0700");

        // Socket file exists once bound.
        assert!(socket_path.exists(), "socket file should exist after bind");

        let (tx, rx) = oneshot::channel::<()>();
        let serve_path = socket_path.clone();
        let server = tokio::spawn(async move {
            bound
                .serve_until(async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
            // Listener dropped here -> socket removed.
            drop(serve_path);
        });

        // Connect and exercise ping -> pong.
        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        stream.write_all(b"ping\n").await.expect("write ping");
        stream.flush().await.expect("flush ping");

        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).await.expect("read pong");
        assert_eq!(&buf, b"pong\n");

        // Trigger shutdown and wait for the server task to finish.
        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");

        // Socket file must be gone after shutdown (listener drop).
        assert!(
            !socket_path.exists(),
            "socket file should be removed after shutdown"
        );

        // Cleanup temp dir.
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- FIX 1: pre-bind ownership guard ----

    #[test]
    fn bind_refuses_when_socket_path_is_a_symlink() {
        // An attacker plants a symlink at the socket path. `bind` must refuse
        // before unlinking/binding through it.
        let dir = temp_socket_dir();
        std::fs::create_dir_all(&dir).expect("create socket dir");
        let socket_path = dir.join("nark.sock");
        let target = dir.join("elsewhere.sock");
        std::fs::write(&target, b"").expect("create symlink target");
        std::os::unix::fs::symlink(&target, &socket_path).expect("plant symlink");

        // `BoundListener` is not `Debug`, so match rather than `expect_err`.
        let err = match BoundListener::bind(&socket_path) {
            Ok(_) => panic!("bind must refuse a symlinked socket path"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("symlink"),
            "rejection should name the symlink-swap case, got: {err:#}"
        );
        // The symlink must be left in place (not unlinked) since we refused.
        assert!(
            std::fs::symlink_metadata(&socket_path)
                .expect("symlink should still exist")
                .file_type()
                .is_symlink(),
            "guard must refuse before unlinking the planted symlink"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn bind_succeeds_for_normal_cold_start() {
        // No pre-existing path: a normal start binds successfully.
        // (`UnixListener::bind` needs a tokio reactor, hence the async test.)
        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("cold-start bind should succeed");
        assert!(socket_path.exists(), "socket file should exist after bind");
        assert_eq!(bound.path(), socket_path.as_path());
        drop(bound); // removes the socket file
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- FIX 2: dedicated socket dir, vault root untouched ----

    #[tokio::test]
    async fn ensure_socket_dir_chmods_run_dir_not_vault_root() {
        // Simulate the default layout: vault_dir/run/nark.sock. The 0700 chmod
        // must land on run/, and the vault root's mode must be left alone.
        let vault = temp_socket_dir();
        std::fs::create_dir_all(&vault).expect("create vault dir");
        // Give the vault root a recognizable, non-0700 mode.
        std::fs::set_permissions(&vault, std::fs::Permissions::from_mode(0o755))
            .expect("set vault mode");
        let vault_mode_before = std::fs::metadata(&vault)
            .expect("stat vault")
            .permissions()
            .mode()
            & 0o777;

        let socket_path = resolve_socket_path(&vault, None);
        assert!(
            socket_path.ends_with("run/nark.sock"),
            "default socket should live under run/, got {}",
            socket_path.display()
        );

        let bound = BoundListener::bind(&socket_path).expect("bind under run/");

        // run/ must be 0700.
        let run_dir = socket_path.parent().expect("run dir");
        let run_mode = std::fs::metadata(run_dir)
            .expect("stat run dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(run_mode, 0o700, "run/ should be chmodded to 0700");

        // Vault root mode must be unchanged.
        let vault_mode_after = std::fs::metadata(&vault)
            .expect("stat vault after")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            vault_mode_after, vault_mode_before,
            "vault root mode must not be modified by serve"
        );
        assert_ne!(
            vault_mode_after, 0o700,
            "vault root should retain its original (non-0700) mode"
        );

        drop(bound);
        let _ = std::fs::remove_dir_all(&vault);
    }

    // ---- FIX 3: per-connection read timeout ----

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn authenticated_connection_with_no_request_is_closed_after_timeout() {
        use super::super::AgentMap;
        use std::collections::HashMap;

        let dir = temp_socket_dir();
        let socket_path = dir.join("nark.sock");
        let bound = BoundListener::bind(&socket_path).expect("bind listener");

        // Authenticate the test process as a known agent.
        let me = nix::unistd::getuid().as_raw();
        let mut table = HashMap::new();
        table.insert(me, "tester".to_string());
        let agents = AgentMap::new(table);

        // Short timeout so the test does not hang.
        let read_timeout = Duration::from_millis(150);

        let (tx, rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            bound
                .serve_authenticated_until_with_timeout(agents, read_timeout, async {
                    let _ = rx.await;
                })
                .await
                .expect("serve loop");
        });

        let mut stream = UnixStream::connect(&socket_path)
            .await
            .expect("connect to socket");
        // Deliberately send nothing (no newline). The server must time out the
        // read and close the connection, so our next read sees EOF.
        let mut buf = Vec::new();
        let read = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut buf))
            .await
            .expect("server should close the idle connection within the test budget")
            .expect("read to EOF");
        assert_eq!(read, 0, "idle connection should be closed at EOF, no bytes");

        tx.send(()).expect("send shutdown");
        server.await.expect("server task join");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
