#!/usr/bin/env bash
set -euo pipefail

REPO="christopher-kapic/mixer-rs"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BINARY="mixer"

CLEANUP_DIR=""
trap 'rm -rf "$CLEANUP_DIR"' EXIT

main() {
  local version="${1:-latest}"

  detect_platform
  resolve_version "$version"

  echo "Installing ${BINARY} ${VERSION} for ${TARGET}..."

  CLEANUP_DIR="$(mktemp -d)"
  local tmp="$CLEANUP_DIR"

  local tarball="${BINARY}-${TARGET}.tar.gz"
  local url="https://github.com/${REPO}/releases/download/${VERSION}/${tarball}"
  local checksum_url="${url}.sha256"

  echo "Downloading ${url}"
  curl -fSL --progress-bar -o "${tmp}/${tarball}" "$url"

  echo "Verifying checksum..."
  curl -fsSL -o "${tmp}/${tarball}.sha256" "$checksum_url"
  (cd "$tmp" && verify_checksum "${tarball}")

  tar xzf "${tmp}/${tarball}" -C "$tmp"

  install_binary "$tmp"

  echo "Installed ${BINARY} ${VERSION} to ${INSTALL_DIR}/${BINARY}"
}

detect_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux)  os="unknown-linux-gnu" ;;
    Darwin) os="apple-darwin" ;;
    *)      echo "Unsupported OS: ${os}" >&2; exit 1 ;;
  esac

  case "$arch" in
    x86_64)        arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *)             echo "Unsupported architecture: ${arch}" >&2; exit 1 ;;
  esac

  TARGET="${arch}-${os}"
}

resolve_version() {
  if [ "$1" = "latest" ]; then
    VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
      | grep '"tag_name"' | head -1 | cut -d'"' -f4)"
    if [ -z "$VERSION" ]; then
      echo "Failed to determine latest version" >&2
      exit 1
    fi
  else
    VERSION="$1"
  fi
}

verify_checksum() {
  local file="$1"
  if command -v sha256sum &>/dev/null; then
    sha256sum --check "${file}.sha256"
  elif command -v shasum &>/dev/null; then
    shasum -a 256 --check "${file}.sha256"
  else
    echo "Warning: no checksum tool found, skipping verification" >&2
  fi
}

install_binary() {
  local src="$1/${BINARY}"
  chmod +x "$src"

  if [ -w "$INSTALL_DIR" ]; then
    mv "$src" "${INSTALL_DIR}/${BINARY}"
  else
    echo "Writing to ${INSTALL_DIR} requires elevated permissions."
    sudo mv "$src" "${INSTALL_DIR}/${BINARY}"
  fi
}

main "$@"
