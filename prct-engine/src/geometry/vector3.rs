// 3D vector operations for geometric calculations
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};

/// 3D vector with double precision coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    /// Create new vector
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3 { x, y, z }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Vector3::new(0.0, 0.0, 0.0)
    }

    /// Unit vectors
    pub fn unit_x() -> Self { Vector3::new(1.0, 0.0, 0.0) }
    pub fn unit_y() -> Self { Vector3::new(0.0, 1.0, 0.0) }
    pub fn unit_z() -> Self { Vector3::new(0.0, 0.0, 1.0) }

    /// Vector magnitude (Euclidean norm)
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Squared magnitude (avoids sqrt for performance)
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Distance to another point
    pub fn distance(&self, other: &Vector3) -> f64 {
        (*self - *other).magnitude()
    }

    /// Squared distance (avoids sqrt for performance)
    pub fn distance_squared(&self, other: &Vector3) -> f64 {
        (*self - *other).magnitude_squared()
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Vector3 {
        let mag = self.magnitude();
        if mag > f64::EPSILON {
            *self / mag
        } else {
            Vector3::zero()
        }
    }

    /// Dot product
    pub fn dot(&self, other: &Vector3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product
    pub fn cross(&self, other: &Vector3) -> Vector3 {
        Vector3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Check if vector is finite (no NaN or infinity)
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Check if vector is approximately equal within tolerance
    pub fn approx_eq(&self, other: &Vector3, tolerance: f64) -> bool {
        (self.x - other.x).abs() < tolerance &&
        (self.y - other.y).abs() < tolerance &&
        (self.z - other.z).abs() < tolerance
    }

    /// Angle between two vectors (in radians)
    pub fn angle_between(&self, other: &Vector3) -> f64 {
        let dot = self.dot(other);
        let mag_product = self.magnitude() * other.magnitude();
        
        if mag_product > f64::EPSILON {
            (dot / mag_product).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        }
    }

    /// Project this vector onto another vector
    pub fn project_onto(&self, other: &Vector3) -> Vector3 {
        let other_mag_sq = other.magnitude_squared();
        if other_mag_sq > f64::EPSILON {
            *other * (self.dot(other) / other_mag_sq)
        } else {
            Vector3::zero()
        }
    }

    /// Component-wise minimum
    pub fn component_min(&self, other: &Vector3) -> Vector3 {
        Vector3::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Component-wise maximum
    pub fn component_max(&self, other: &Vector3) -> Vector3 {
        Vector3::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    /// Linear interpolation between two vectors
    pub fn lerp(&self, other: &Vector3, t: f64) -> Vector3 {
        *self * (1.0 - t) + *other * t
    }
}

/// Calculate dihedral angle between four points (in radians)
/// Uses the IUPAC convention: angle from 0 to π is positive, from 0 to -π is negative
pub fn calculate_dihedral_angle(p1: &Vector3, p2: &Vector3, p3: &Vector3, p4: &Vector3) -> f64 {
    // Vector from p2 to p1
    let b1 = *p1 - *p2;
    // Vector from p2 to p3  
    let b2 = *p3 - *p2;
    // Vector from p3 to p4
    let b3 = *p4 - *p3;
    
    // Cross products
    let c1 = b1.cross(&b2);
    let c2 = b2.cross(&b3);
    
    // Magnitudes
    let c1_mag = c1.magnitude();
    let c2_mag = c2.magnitude();
    
    if c1_mag < f64::EPSILON || c2_mag < f64::EPSILON {
        return 0.0; // Degenerate case
    }
    
    // Normalized cross products
    let n1 = c1 / c1_mag;
    let n2 = c2 / c2_mag;
    
    // Dot product for cosine
    let cos_angle = n1.dot(&n2).clamp(-1.0, 1.0);
    
    // Cross product for sign
    let cross = n1.cross(&n2);
    let sin_sign = cross.dot(&(b2 / b2.magnitude()));
    
    let angle = cos_angle.acos();
    
    if sin_sign < 0.0 {
        -angle
    } else {
        angle
    }
}

/// Geometric functions using Vector3
impl Vector3 {
    /// Calculate dihedral angle between four points
    /// Returns angle in radians [-π, π]
    pub fn dihedral_angle(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3) -> f64 {
        // Bond vectors
        let b1 = p2 - p1;
        let b2 = p3 - p2;
        let b3 = p4 - p3;
        
        // Normal vectors to planes
        let n1 = b1.cross(&b2);
        let n2 = b2.cross(&b3);
        
        // Check for linear configurations (epsilon for numerical stability)
        let epsilon = 1e-10;
        if n1.magnitude() < epsilon || n2.magnitude() < epsilon {
            return 0.0; // Undefined dihedral for linear arrangement
        }
        
        // Normalize normal vectors
        let n1_norm = n1.normalize();
        let n2_norm = n2.normalize();
        
        // Calculate dihedral angle using atan2 for correct quadrant
        let cos_angle = n1_norm.dot(&n2_norm);
        let sin_angle = (n1_norm.cross(&n2_norm)).dot(&b2.normalize());
        
        sin_angle.atan2(cos_angle)
    }

    /// Calculate bond angle between three points (in radians)
    pub fn bond_angle(p1: Vector3, p2: Vector3, p3: Vector3) -> f64 {
        let v1 = (p1 - p2).normalize();
        let v2 = (p3 - p2).normalize();
        v1.angle_between(&v2)
    }

    /// Check if four points are approximately coplanar
    pub fn are_coplanar(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3, tolerance: f64) -> bool {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let v3 = p4 - p1;
        
        // Volume of tetrahedron formed by the points
        let volume = v1.dot(&v2.cross(&v3)).abs() / 6.0;
        volume < tolerance
    }
}

// Operator implementations
impl Add for Vector3 {
    type Output = Vector3;
    fn add(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vector3 {
    type Output = Vector3;
    fn sub(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;
    fn mul(self, scalar: f64) -> Vector3 {
        Vector3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vector3> for f64 {
    type Output = Vector3;
    fn mul(self, vector: Vector3) -> Vector3 {
        vector * self
    }
}

impl Div<f64> for Vector3 {
    type Output = Vector3;
    fn div(self, scalar: f64) -> Vector3 {
        Vector3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl Neg for Vector3 {
    type Output = Vector3;
    fn neg(self) -> Vector3 {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, other: Vector3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl SubAssign for Vector3 {
    fn sub_assign(&mut self, other: Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl MulAssign<f64> for Vector3 {
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl DivAssign<f64> for Vector3 {
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

impl fmt::Display for Vector3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

impl From<[f64; 3]> for Vector3 {
    fn from(coords: [f64; 3]) -> Self {
        Vector3::new(coords[0], coords[1], coords[2])
    }
}

impl From<(f64, f64, f64)> for Vector3 {
    fn from(coords: (f64, f64, f64)) -> Self {
        Vector3::new(coords.0, coords.1, coords.2)
    }
}

impl Into<[f64; 3]> for Vector3 {
    fn into(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_basic_operations() {
        let v1 = Vector3::new(1.0, 2.0, 3.0);
        let v2 = Vector3::new(4.0, 5.0, 6.0);

        assert_eq!(v1 + v2, Vector3::new(5.0, 7.0, 9.0));
        assert_eq!(v2 - v1, Vector3::new(3.0, 3.0, 3.0));
        assert_eq!(v1 * 2.0, Vector3::new(2.0, 4.0, 6.0));
        assert_eq!(v1 / 2.0, Vector3::new(0.5, 1.0, 1.5));
    }

    #[test]
    fn test_vector_magnitude() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        assert!((v.magnitude() - 5.0).abs() < f64::EPSILON);
        assert!((v.magnitude_squared() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new(0.0, 1.0, 0.0);
        assert!((v1.dot(&v2) - 0.0).abs() < f64::EPSILON);

        let v3 = Vector3::new(1.0, 1.0, 0.0);
        assert!((v1.dot(&v3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cross_product() {
        let v1 = Vector3::new(1.0, 0.0, 0.0);
        let v2 = Vector3::new(0.0, 1.0, 0.0);
        let cross = v1.cross(&v2);
        assert!(cross.approx_eq(&Vector3::new(0.0, 0.0, 1.0), 1e-10));
    }

    #[test]
    fn test_dihedral_angle() {
        // Test case for perfect tetrahedral angle
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(1.0, 0.0, 0.0);
        let p3 = Vector3::new(1.5, 0.866, 0.0); // 60° angle
        let p4 = Vector3::new(1.5, 0.289, 0.816); // Tetrahedral

        let dihedral = Vector3::dihedral_angle(p1, p2, p3, p4);
        
        // Should be non-zero for non-coplanar arrangement
        assert!(dihedral.abs() > 1e-10);
        assert!(dihedral.abs() <= std::f64::consts::PI);
    }

    #[test]
    fn test_normalize() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
        assert!(normalized.approx_eq(&Vector3::new(0.6, 0.8, 0.0), 1e-10));
    }
}