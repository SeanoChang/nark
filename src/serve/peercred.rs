//! Peer-credential extraction for `nark serve` connections.
//!
//! Slice 2.3 lands the macOS path: `getpeereid(2)` over the connected Unix
//! socket yields the peer's effective uid, which the daemon uses to
//! authenticate the calling agent against the owning user.
//!
//! Linux peer credentials (`SO_PEERCRED`) are deferred to a production Linux
//! slice; the Linux arm returns an explicit error so callers fail closed.

#[cfg(target_os = "macos")]
use tokio::net::UnixStream;

/// Extract the peer's uid from a connected Unix domain socket.
///
/// On macOS this calls `getpeereid(2)` on the socket file descriptor. On Linux
/// this is not yet implemented and returns an error so the caller fails closed.
#[cfg(target_os = "macos")]
// Wired into the accept path in a later slice; tests exercise it today.
#[cfg_attr(not(test), allow(dead_code))]
pub fn peer_uid(stream: &UnixStream) -> anyhow::Result<u32> {
    use std::os::fd::AsFd;

    let (uid, _gid) = nix::unistd::getpeereid(stream.as_fd())
        .map_err(|e| anyhow::anyhow!("getpeereid failed: {e}"))?;
    Ok(uid.as_raw())
}

/// Linux stub: `SO_PEERCRED`-based peer credentials are not yet implemented.
#[cfg(target_os = "linux")]
// Wired into the accept path in a later slice; tests exercise it today.
#[cfg_attr(not(test), allow(dead_code))]
pub fn peer_uid(_stream: &tokio::net::UnixStream) -> anyhow::Result<u32> {
    Err(anyhow::anyhow!(
        "SO_PEERCRED not yet implemented (Linux) — TODO Phase: prod Linux"
    ))
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn peer_uid_matches_current_user() {
        use super::peer_uid;
        use tokio::net::{UnixListener, UnixStream};

        let dir = std::env::temp_dir().join(format!(
            "nark-peercred-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let socket_path = dir.join("peer.sock");

        let listener = UnixListener::bind(&socket_path).expect("bind listener");

        let connect_path = socket_path.clone();
        let client = tokio::spawn(async move {
            UnixStream::connect(&connect_path)
                .await
                .expect("client connect")
        });

        let (server_stream, _addr) = listener.accept().await.expect("accept connection");
        let _client_stream = client.await.expect("client task join");

        let extracted = peer_uid(&server_stream).expect("extract peer uid");
        let expected = nix::unistd::getuid().as_raw();
        assert_eq!(
            extracted, expected,
            "peer uid should match the connecting (current) user"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "Linux SO_PEERCRED peer-uid extraction not yet implemented"]
    fn peer_uid_linux_placeholder() {
        // Placeholder: Linux peer-credential extraction lands in a later slice.
    }
}
