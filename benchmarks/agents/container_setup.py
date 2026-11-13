"""In-container agent installation that survives a read-only image.

The agents this suite drives are npm packages, but the images they run in are
``.sif`` files: read-only squashfs plus a small (64 MiB) writable overlay. Two
consequences decide how the install has to be written:

* ``apt-get install nodejs`` fails, because dpkg tries to move an existing
  directory into the overlay and overlayfs answers ``Invalid cross-device link``;
* ``npm install -g`` needs ~300 MiB, far past the 64 MiB overlay, so the tools
  must live on a directory bind-mounted from the host (``/opt/agent``) rather
  than inside the image. Only the symlinks into ``/usr/local/bin`` are written
  to the overlay, which they fit in.

A bare ``curl`` is also frequently absent from slim images, so the download
falls back to ``wget`` and then to ``python3``.
"""

#: Where the suite bind-mounts a host directory for agent tools.
PREFIX = "/opt/agent"

NODE_VERSION = "22.14.0"


def node_commands(version=NODE_VERSION, prefix=PREFIX):
    """Shell commands that put a usable ``node``/``npm`` on PATH."""
    return [
        "set -euo pipefail; "
        'arch="$(uname -m)"; '
        'if [ "$arch" = "x86_64" ] || [ "$arch" = "amd64" ]; then nodearch=linux-x64; '
        'elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then nodearch=linux-arm64; '
        'else echo "unsupported architecture: $arch" >&2; exit 1; fi; '
        f"mkdir -p {prefix} && cd /tmp && "
        f'url="https://nodejs.org/dist/v{version}/node-v{version}-${{nodearch}}.tar.gz"; '
        'if command -v curl >/dev/null 2>&1; then curl -fsSL "$url" -o node.tar.gz; '
        'elif command -v wget >/dev/null 2>&1; then wget -qO node.tar.gz "$url"; '
        "else python3 -c 'import sys, urllib.request; "
        'urllib.request.urlretrieve(sys.argv[1], "node.tar.gz")\' "$url"; fi; '
        f"tar -xzf node.tar.gz -C {prefix} --strip-components=1 && rm -f node.tar.gz && "
        f"ln -sf {prefix}/bin/node /usr/local/bin/node && "
        f"ln -sf {prefix}/bin/npm /usr/local/bin/npm && "
        f"ln -sf {prefix}/bin/npx /usr/local/bin/npx && "
        f"{prefix}/bin/node --version"
    ]


def npm_global_commands(package, executable, version=None, prefix=PREFIX):
    """Install an npm package globally and expose ``executable`` on PATH.

    The npm cache is redirected onto the bind mount as well: the default
    ``~/.npm`` would otherwise fill the image's 64 MiB overlay on its own.
    """
    spec = f"{package}@{version}" if version else package
    return [
        f"{prefix}/bin/npm install -g --no-fund --no-audit "
        f"--cache {prefix}/npm-cache --prefix {prefix} {spec}",
        f"ln -sf {prefix}/bin/{executable} /usr/local/bin/{executable}",
        f"{prefix}/bin/{executable} --version",
    ]
