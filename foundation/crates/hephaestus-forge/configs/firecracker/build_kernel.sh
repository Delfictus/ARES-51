#!/bin/bash
# Build minimal Linux kernel for Firecracker VMs
# Optimized for <50ms boot time and maximum security

set -euo pipefail

KERNEL_VERSION="6.1.70"
KERNEL_DIR="/tmp/linux-${KERNEL_VERSION}"
OUTPUT_DIR="/opt/forge/minimal_kernel"
CONFIG_FILE="$(dirname "$0")/minimal_kernel_config"

echo "Building minimal Linux kernel v${KERNEL_VERSION} for Firecracker..."

# Create output directory
sudo mkdir -p "${OUTPUT_DIR}"

# Download kernel source if not exists
if [ ! -d "${KERNEL_DIR}" ]; then
    echo "Downloading Linux kernel v${KERNEL_VERSION}..."
    cd /tmp
    wget -q "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-${KERNEL_VERSION}.tar.xz"
    tar -xf "linux-${KERNEL_VERSION}.tar.xz"
    rm "linux-${KERNEL_VERSION}.tar.xz"
fi

cd "${KERNEL_DIR}"

# Apply our minimal configuration
echo "Applying minimal security-hardened configuration..."
cp "${CONFIG_FILE}" .config

# Make sure config is valid and apply defaults for new options
make olddefconfig

# Build kernel with maximum parallelism
echo "Building kernel (this may take 5-10 minutes)..."
make -j$(nproc) bzImage

# Copy kernel to output directory
echo "Installing kernel to ${OUTPUT_DIR}..."
sudo cp arch/x86/boot/bzImage "${OUTPUT_DIR}/vmlinux"
sudo chmod 644 "${OUTPUT_DIR}/vmlinux"

# Create minimal initramfs for emergency cases
echo "Creating minimal initramfs..."
INITRAMFS_DIR="/tmp/initramfs"
rm -rf "${INITRAMFS_DIR}"
mkdir -p "${INITRAMFS_DIR}"/{bin,sbin,etc,proc,sys,dev,tmp}

# Copy essential binaries (if available)
if command -v busybox >/dev/null 2>&1; then
    cp $(which busybox) "${INITRAMFS_DIR}/bin/"
    cd "${INITRAMFS_DIR}"
    # Create symlinks for common commands
    for cmd in sh ls cat echo mount umount; do
        ln -sf /bin/busybox "bin/$cmd"
        ln -sf /bin/busybox "sbin/$cmd"
    done
fi

# Create minimal init script
cat > "${INITRAMFS_DIR}/init" << 'EOF'
#!/bin/sh
# Minimal init for Firecracker emergency boot
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
echo "Minimal init loaded - system ready"
exec /bin/sh
EOF

chmod +x "${INITRAMFS_DIR}/init"

# Create initramfs archive
cd "${INITRAMFS_DIR}"
find . | cpio -o -H newc | gzip > "${OUTPUT_DIR}/initramfs.gz"

echo "Kernel build complete!"
echo "Kernel: ${OUTPUT_DIR}/vmlinux"
echo "Initramfs: ${OUTPUT_DIR}/initramfs.gz"
echo ""
echo "Kernel size: $(du -h ${OUTPUT_DIR}/vmlinux | cut -f1)"
echo "Expected boot time: <50ms in Firecracker VM"
echo ""
echo "Security features enabled:"
echo "  - Hardware isolation via KVM"
echo "  - No loadable modules"
echo "  - Minimal attack surface"
echo "  - KASLR and stack protection"
echo "  - Page poisoning and SLAB hardening"
echo "  - Seccomp filter support"
echo "  - No 32-bit compatibility"
echo "  - No container/namespace support (Firecracker provides isolation)"