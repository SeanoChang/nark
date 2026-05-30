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
//! Stale-socket handling is a simple unlink-if-exists for now; full liveness
//! probing lands in a later slice.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use super::AgentMap;
use super::peercred::peer_uid;

/// Resolve the socket path for the daemon.
///
/// When `socket` is provided it is used verbatim; otherwise the path defaults
/// to `<vault_dir>/nark.sock`.
pub fn resolve_socket_path(vault_dir: &Path, socket: Option<String>) -> PathBuf {
    match socket {
        Some(s) => PathBuf::from(s),
        None => vault_dir.join("nark.sock"),
    }
}

/// Ensure the parent directory of `socket_path` exists at mode `0700`.
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
    /// Bind a fresh listener at `socket_path`, creating the `0700` parent
    /// directory and unlinking any stale socket file first.
    pub fn bind(socket_path: &Path) -> Result<Self> {
        ensure_socket_dir(socket_path)?;
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
    pub async fn serve_authenticated_until<F>(&self, agents: AgentMap, shutdown: F) -> Result<()>
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
                        if let Err(e) = handle_authenticated_connection(stream, &agents).await {
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
async fn handle_authenticated_connection(stream: UnixStream, agents: &AgentMap) -> Result<()> {
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
    fn resolve_defaults_to_vault_nark_sock() {
        let vault = Path::new("/tmp/some-vault");
        let resolved = resolve_socket_path(vault, None);
        assert_eq!(resolved, Path::new("/tmp/some-vault/nark.sock"));
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
}
