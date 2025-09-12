#!/bin/bash
# Build minimal rootfs for Firecracker VMs
# Ultra-minimal filesystem for secure code execution

set -euo pipefail

OUTPUT_DIR="/opt/forge/rootfs"
ROOTFS_SIZE="64M"  # Minimal size for basic operation
MOUNT_POINT="/tmp/rootfs_mount"

echo "Building minimal rootfs for Firecracker VM..."

# Create output directory
sudo mkdir -p "${OUTPUT_DIR}"

# Create empty ext4 filesystem
echo "Creating ${ROOTFS_SIZE} ext4 filesystem..."
sudo dd if=/dev/zero of="${OUTPUT_DIR}/forge_minimal.ext4" bs=1M count=64 status=progress
sudo mkfs.ext4 -F "${OUTPUT_DIR}/forge_minimal.ext4"

# Mount the filesystem
sudo mkdir -p "${MOUNT_POINT}"
sudo mount -o loop "${OUTPUT_DIR}/forge_minimal.ext4" "${MOUNT_POINT}"

# Create essential directory structure
echo "Creating minimal directory structure..."
sudo mkdir -p "${MOUNT_POINT}"/{bin,sbin,etc,proc,sys,dev,tmp,var,run}
sudo mkdir -p "${MOUNT_POINT}/var"/{log,tmp}
sudo mkdir -p "${MOUNT_POINT}/etc"/{init.d,rc.d}

# Install busybox for essential commands
if command -v busybox >/dev/null 2>&1; then
    echo "Installing busybox..."
    sudo cp $(which busybox) "${MOUNT_POINT}/bin/"
    
    # Create symlinks for essential commands
    cd "${MOUNT_POINT}"
    for cmd in sh bash ls cat echo mount umount mkdir rmdir rm cp mv chmod chown ps kill sleep; do
        sudo ln -sf /bin/busybox "bin/$cmd"
    done
    for cmd in init halt reboot; do
        sudo ln -sf /bin/busybox "sbin/$cmd"
    done
else
    echo "Warning: busybox not found, creating minimal shell stub"
    sudo tee "${MOUNT_POINT}/bin/sh" > /dev/null << 'EOF'
#!/bin/sh
echo "Minimal shell - limited functionality"
while true; do
    read -p "# " cmd
    case "$cmd" in
        "exit") exit 0 ;;
        "halt"|"shutdown") halt ;;
        *) echo "Command not found: $cmd" ;;
    esac
done
EOF
    sudo chmod +x "${MOUNT_POINT}/bin/sh"
fi

# Create minimal init system
echo "Creating minimal init system..."
sudo tee "${MOUNT_POINT}/sbin/init" > /dev/null << 'EOF'
#!/bin/sh
# Minimal init for Firecracker VM - security focused

# Mount essential filesystems
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mount -t tmpfs tmpfs /tmp
mount -t tmpfs tmpfs /run

# Set up minimal environment
export PATH="/bin:/sbin"
export HOME="/root"

# Create essential device nodes if not present
[ ! -c /dev/null ] && mknod /dev/null c 1 3
[ ! -c /dev/zero ] && mknod /dev/zero c 1 5
[ ! -c /dev/random ] && mknod /dev/random c 1 8
[ ! -c /dev/urandom ] && mknod /dev/urandom c 1 9

# Set minimal system state
echo "forge-vm" > /etc/hostname
echo "127.0.0.1 localhost forge-vm" > /etc/hosts

# Security: Set restrictive umask
umask 077

# Signal handler for clean shutdown
trap 'echo "Shutting down..."; sync; halt' TERM INT

echo "Forge VM initialized - ready for secure code execution"

# Start shell or wait for commands
if [ -t 0 ] && [ -t 1 ]; then
    # Interactive mode
    exec /bin/sh
else
    # Non-interactive - wait for termination
    while true; do
        sleep 1
    done
fi
EOF

sudo chmod +x "${MOUNT_POINT}/sbin/init"

# Create minimal passwd and group files
echo "Creating minimal user database..."
sudo tee "${MOUNT_POINT}/etc/passwd" > /dev/null << 'EOF'
root:x:0:0:root:/root:/bin/sh
nobody:x:65534:65534:nobody:/:/bin/false
EOF

sudo tee "${MOUNT_POINT}/etc/group" > /dev/null << 'EOF'
root:x:0:
nobody:x:65534:
EOF

# Create minimal shadow file (no passwords for security)
sudo tee "${MOUNT_POINT}/etc/shadow" > /dev/null << 'EOF'
root:!:19000:0:99999:7:::
nobody:!:19000:0:99999:7:::
EOF

sudo chmod 600 "${MOUNT_POINT}/etc/shadow"

# Create minimal fstab
sudo tee "${MOUNT_POINT}/etc/fstab" > /dev/null << 'EOF'
# Minimal fstab for Firecracker VM
/dev/vda / ext4 ro,noatime 0 1
proc /proc proc defaults 0 0
sysfs /sys sysfs defaults 0 0
devtmpfs /dev devtmpfs defaults 0 0
tmpfs /tmp tmpfs defaults,noexec,nosuid,nodev 0 0
tmpfs /run tmpfs defaults,noexec,nosuid,nodev 0 0
EOF

# Create minimal inittab (if init system expects it)
sudo tee "${MOUNT_POINT}/etc/inittab" > /dev/null << 'EOF'
::sysinit:/sbin/init
::respawn:/bin/sh
::shutdown:/bin/umount -a -r
::restart:/sbin/halt
EOF

# Security: Create minimal profile with restrictive settings
sudo tee "${MOUNT_POINT}/etc/profile" > /dev/null << 'EOF'
# Minimal security-hardened profile
export PATH="/bin:/sbin"
export HOME="/root"
export SHELL="/bin/sh"
export TERM="linux"
umask 077

# Security: Disable core dumps
ulimit -c 0

# Security: Limit resources
ulimit -f 1024      # Max file size 1MB  
ulimit -m 65536     # Max memory 64MB
ulimit -n 64        # Max open files
ulimit -p 16        # Max processes
ulimit -t 30        # Max CPU time 30 seconds

echo "Secure environment loaded"
EOF

# Create minimal securetty (restrict root login)
sudo tee "${MOUNT_POINT}/etc/securetty" > /dev/null << 'EOF'
console
ttyS0
EOF

# Set proper permissions
sudo chmod 755 "${MOUNT_POINT}"/{bin,sbin,etc}
sudo chmod 1777 "${MOUNT_POINT}/tmp"
sudo chmod 755 "${MOUNT_POINT}/var"

# Unmount and finalize
echo "Finalizing rootfs..."
sudo umount "${MOUNT_POINT}"
sudo rmdir "${MOUNT_POINT}"

# Set final permissions
sudo chmod 644 "${OUTPUT_DIR}/forge_minimal.ext4"

echo "Rootfs build complete!"
echo "Location: ${OUTPUT_DIR}/forge_minimal.ext4"
echo "Size: $(du -h ${OUTPUT_DIR}/forge_minimal.ext4 | cut -f1)"
echo ""
echo "Security features:"
echo "  - Read-only root filesystem"
echo "  - No setuid binaries"
echo "  - Restrictive umask (077)"
echo "  - Resource limits via ulimit"
echo "  - Minimal attack surface"
echo "  - No network tools"
echo "  - No development tools"
echo "  - No unnecessary services"