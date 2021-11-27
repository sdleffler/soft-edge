//! # Soft Edge: efficient bithackery for making 3D collision meshes out of grids and stacked tile maps
//!
//! *indecipherable wailing guitar noises and tape echo sounds* - Wata (Boris - Soft Edge)
//!
//! `soft-edge` is a crate which provides utilities for dealing with 3D grids where the cells are
//! colliders defined as convex hulls of subsets of the unit cube.
//!
//! Things it gives you:
//! - [`Vertex`], representing a vertex of the unit cube as an index in `0..8`
//! - [`VertexSet`], representing a subset of the vertices of the unit cube as a 1-byte bitset
//! - [`Atom`], representing a valid convex non-coplanar subset of the vertices of the unit cube
//! - [`CompoundHull`], representing a clippable convex hull of an `Atom`
//! - [`HullFacet`], representing a polygon of a potentially clipped compound hull of an atom.
//!   Facets calculated from a [`CompoundHull`] will *always* be in CCW winding order.
//!
//! # Why?
//!
//! If you've ever tried building a game physics system with a tile-based map before, you've
//! probably run into the "crack problem". This is where if you have two adjacent tiles, things
//! which should smoothly move over the "crack" between them (which mathematically does not exist
//! because the vertices of these two tiles are shared) instead get caught on the crack, because
//! during collision detection the object is moved into a tile by some force and ends up spuriously
//! colliding with an unexposed edge of one tile.
//!
//! `soft-edge`'s [`CompoundHull`] type represents a sort of encoded tile collider with a
//! `join_exteriors` method that erases any of these unexposed edges between two tiles, leaving you
//! with the surface polygons that can't cause the crack problem. In addition, these clipping
//! operations are extremely fast, requiring only three bitwise operations (maybe some more because
//! they're done on only a subset of the bits of a thing, but still, it's fast.)
//!
//! # Does it work?
//!
//! Uh, I think so. All the tests pass, at least?
//!
//! Many things in this crate are specialized/brute forced/handwritten as lookup tables. I've fixed
//! all the bugs I could find, but I'm sure there are more hiding somewhere, probably in mistakes in
//! bit-significance or something. Though most bitwise ops are done through the `bitvec` crate
//! exactly to avoid this as much as possible.
//!
//! # Vertex numbering
//!
//! Vertices are numbered as follows:
//!
//! ```text
//!   v3      v7
//!     *----*
//! v2 /| v6/|
//!   *----* |   +Y
//!   | *--|-*   ^ ^ +Z
//!   |/v1 |/v5  |/
//!   *----*     +--> +X
//!  v0    v4
//! ```
//!
//! For more convenience, a conversion between vertex index and unit cube coordinates:
//!
//! ```text
//! // v0 is the origin.
//! v0 <=> {0, 0, 0} <=> 0b000
//! v1 <=> {0, 0, 1} <=> 0b001
//! v2 <=> {0, 1, 0} <=> 0b010
//! v3 <=> {0, 1, 1} <=> 0b011
//! v4 <=> {1, 0, 0} <=> 0b100
//! v5 <=> {1, 0, 1} <=> 0b101
//! v6 <=> {1, 1, 0} <=> 0b110
//! v7 <=> {1, 1, 1} <=> 0b111
//! ```
//!
//! This is derived as a bitwise representation `0bXYZ` where the X bit represents whether or not
//! the X axis is `1`, Y bit is whether or not the Y axis is `1`, etc.
//!
//! # "Exact" coordinates for vertex index deduplication
//!
//! In order to facilitate vertex deduplication (for an indexed triangle mesh, for example) most of
//! the output coordinates for things like the [`CompoundHull`]'s facets ([`HullFacet`]) are
//! presented using the [`Exact`] type. This wraps nalgebra's [`Point3<i32>`], and is essentially
//! just the same coordinates you would expect but scaled by a factor of 2 on every axis. This
//! allows us to represent centroids of faces with no possibility of error, and also provides a
//! hashable type which can be used to build a set mapping vertices to indices.
//!
//! # Generally speaking, how do you expect me to use this?
//!
//! 1. Encode your collision map as a grid of bytes, which correspond to valid [`Atom`]s.
//!    Deserialize these bytes into [`VertexSet`]s using [`VertexSet::from_u8`], and try to
//!    construct atoms from them with [`Atom::try_new`].
//! 2. Construct a three-dimensional grid consisting of the [`CompoundHull`]s of the atom grid
//!    elements, and then for every hull in the grid, `join` it with its neighbors on the proper
//!    axes.
//! 3. Extract facets of these joined compound hulls as [`HullFacet`]s, and then construct your
//!    collision geometry with them. This will end up as a very much non-convex mesh, which you may
//!    want to decompose into convex sub-meshes (but don't have to, if you do collisions directly on
//!    polygons and only allow contact normals which penetrate through their "surface" side.)
//!
//! Step 3 has some caveats. `soft-edge` attempts to produce hull facets with the correct winding
//! order, which should allow you to easily calculate their normals. If it does not, this is a bug,
//! and must be fixed. The caveat here is that `soft-edge` does not yet have a full test suite.
//!

#![warn(missing_docs, missing_debug_implementations)]

use arrayvec::ArrayVec;
use bitvec::prelude::*;
use std::fmt;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign,
};

use nalgebra::*;

mod hull;

pub use bitvec;
pub use hull::*;

/// The eight possible points of a [`Vertex`], in "exact coordinates" (see [`Exact`].)
///
/// [`Vertex::to_exact`] is implemented as a lookup in this array,
pub const POINTS: [Exact; 8] = [
    Exact(point!(0, 0, 0)),
    Exact(point!(0, 0, 2)),
    Exact(point!(0, 2, 0)),
    Exact(point!(0, 2, 2)),
    Exact(point!(2, 0, 0)),
    Exact(point!(2, 0, 2)),
    Exact(point!(2, 2, 0)),
    Exact(point!(2, 2, 2)),
];

/// A plane, represented as a point on the plane and a normal vector (not necessarily guaranteed to
/// be normalized).
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// A point on the plane.
    pub p0: Point3<f32>,
    /// A vector normal to the plane.
    pub n: Vector3<f32>,
}

impl Plane {
    /// Returns `true` if this point is on the same side as its normal ("above" the plane).
    pub fn is_point_above(&self, p1: &Point3<f32>) -> bool {
        (p1 - self.p0).dot(&self.n) > 0.
    }

    /// Returns `true` if this point is on the opposite side as its normal ("below" the plane).
    pub fn is_point_below(&self, p1: &Point3<f32>) -> bool {
        (p1 - self.p0).dot(&self.n) < 0.
    }
}

/// A vertex of a cube, represented as a byte index encoded w/ three bits, each representing an axis
/// as a boolean (true = 1, false = 0.) See the constants [`X_AXIS`], [`Y_AXIS`], [`Z_AXIS`] for a
/// reference of which bits are which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vertex {
    index: u8,
}

impl Vertex {
    /// Cubes have 8 vertices.
    #[inline]
    pub fn is_valid(self) -> bool {
        self.index < 8
    }

    /// Convert this vertex to the corresponding point on the unit cube in "exact" coordinates.
    #[inline]
    pub fn to_exact(self) -> Exact {
        POINTS[self.index as usize]
    }

    /// Convert this vertex to the corresponding point on the unit cube, represented using `f32`
    /// coordinates.
    pub fn to_f32(self) -> Point3<f32> {
        self.to_exact().0.cast() / 2.
    }

    /// Convert this vertex to a `u8` in the range [0, 8].
    #[inline]
    pub fn to_u8(self) -> u8 {
        self.index
    }

    /// Convert a raw bit representation (`u8` in the range [0, 8]) into a `Vertex`. Panics if the
    /// index is invalid (`>= 8`.)
    #[inline]
    pub fn from_u8(byte: u8) -> Self {
        let this = Self { index: byte };
        assert!(this.is_valid(), "invalid vertex");
        this
    }

    /// Construct an iterator which returns all eight possible vertices.
    #[inline]
    pub fn generator() -> impl Iterator<Item = Self> {
        (0..8u8).map(|bits| Self { index: bits })
    }

    /// Convert this vertex to the vertex set containing only it.
    #[inline]
    pub fn to_set(self) -> VertexSet {
        VertexSet::new() | self
    }
}

/// Construct a vertex set in such a way that it can be used in a `const` expression. This macro
/// requires that the `bitarr!` macro and `Lsb0` type from the `bitvec` crate are in scope, for
/// technical reasons related to the limitations of bitvec's `bitarr` macro itself.
#[macro_export]
macro_rules! const_vertex_set {
    ($($val:expr),*) => {
        VertexSet {
            bits: bitarr![const Lsb0, u8; $($val),*],
        }
    }
}

/// Construct a vertex set.
///
/// # Example
///
/// ```
/// # use soft_edge::*;
/// // Construct a vertex set representing a tetrahedron, made with the four vertices of
/// // index 1, 4, 5, and 7.
/// let tetrahedron_set = vertex_set![0, 1, 0, 0, 1, 1, 0, 1];
/// let hull = Atom::new(tetrahedron_set).compound_hull();
/// // A tetrahedron built like this will have one interior face (conceptually, from
/// // slicing off one corner of the cube) and three exterior faces (all triangles,
/// // leftover sides from the slicing operation which used to be squares.)
/// assert_eq!(hull.interior().facets().count(), 1);
/// assert_eq!(hull.exterior().facets().count(), 3,);
/// ```
#[macro_export]
macro_rules! vertex_set {
    ($($val:expr),*) => {
        VertexSet {
            bits: $crate::bitvec::bitarr![$crate::bitvec::order::Lsb0, u8; $($val),*],
        }
    }
}

/// The six faces of the unit cube, each corresponding to an axis.
///
/// This array is arranged such that given some [`Axis`] `axis`, `FACES[axis.to_index()]` will
/// return the appropriate vertex set representing the face in the direction of that axis.
pub const FACES: [VertexSet; 6] = {
    [
        // +X
        const_vertex_set![0, 0, 0, 0, 1, 1, 1, 1],
        // +Y
        const_vertex_set![0, 0, 1, 1, 0, 0, 1, 1],
        // +Z
        const_vertex_set![0, 1, 0, 1, 0, 1, 0, 1],
        // -X
        const_vertex_set![1, 1, 1, 1, 0, 0, 0, 0],
        // -Y
        const_vertex_set![1, 1, 0, 0, 1, 1, 0, 0],
        // -Z
        const_vertex_set![1, 0, 1, 0, 1, 0, 1, 0],
    ]
};

/// Six quadrilaterals which happen to match planes of symmetry of the cube. These are the planes
/// which exist between pairs of edges of the cube, and represent six possible ways we might end up
/// with a quad as an interior face of an atom, plus a total of 24 possible ways we could end up
/// with triangles as interior faces of an atom (by cutting those quads in half, four ways each.)
pub const SYMMETRIES: [VertexSet; 6] = [
    // Symmetry planes

    // 0, 1, 6, 7
    const_vertex_set![1, 1, 0, 0, 0, 0, 1, 1],
    // 2, 3, 4, 5
    const_vertex_set![0, 0, 1, 1, 1, 1, 0, 0],
    // 0, 3, 4, 7
    const_vertex_set![1, 0, 0, 1, 1, 0, 0, 1],
    // 1, 2, 5, 6
    const_vertex_set![0, 1, 1, 0, 0, 1, 1, 0],
    // 1, 3, 4, 6
    const_vertex_set![0, 1, 0, 1, 1, 0, 1, 0],
    // 0, 2, 5, 7
    const_vertex_set![1, 0, 1, 0, 0, 1, 0, 1],
];

/// The eight "anomalous configurations" of subsets of the unit cube; these are triangular interior
/// faces which can be calculated as part of finding the interior convex hull of an atom, and which
/// are always triangular (within an atom, can never be joined with another triangle into a quad.)
/// Also, "anomalous configuration" sounds badass. Admit it.
pub const ANOMALOUS_CONFIGURATIONS: [VertexSet; 8] = [
    // 0, 3, 5
    const_vertex_set![1, 0, 0, 1, 0, 1, 0, 0],
    // 0, 3, 6
    const_vertex_set![1, 0, 0, 1, 0, 0, 1, 0],
    // 0, 5, 6
    const_vertex_set![1, 0, 0, 0, 0, 1, 1, 0],
    // 1, 2, 4
    const_vertex_set![0, 1, 1, 0, 1, 0, 0, 0],
    // 1, 2, 7
    const_vertex_set![0, 1, 1, 0, 0, 0, 0, 1],
    // 1, 4, 7
    const_vertex_set![0, 1, 0, 0, 1, 0, 0, 1],
    // 2, 4, 7
    const_vertex_set![0, 0, 1, 0, 1, 0, 0, 1],
    // 3, 5, 6
    const_vertex_set![0, 0, 0, 1, 0, 1, 1, 0],
];

/// Represents a single signed axis: `+X`, `+Y`, `+Z`, `-X`, `-Y`, or `-Z`.
///
/// These are used to enumerate the faces of the unit cube when deriving the exterior hull of an
/// atom, and are also encoded into the bit representation used by [`Face`].
///
/// Note that on the unit cube, you won't actually find any negative coordinates; the negative axes
/// are used to represent the faces *in the direction of* the negatives, and so are also sometimes
/// called "zero axes" because they represent the face where the corresponding coordinate is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum Axis {
    PosX = 0,
    PosY = 1,
    PosZ = 2,
    NegX = 3,
    NegY = 4,
    NegZ = 5,
}

impl Axis {
    /// # Safety
    ///
    /// This byte must be a valid `Axis` (one of the six byte values defined in the enum.)
    #[inline]
    pub unsafe fn from_u8_unchecked(byte: u8) -> Self {
        std::mem::transmute(byte)
    }

    /// What vertices on our atom are members of the face on this axis?
    #[inline]
    pub fn to_face_set(self) -> VertexSet {
        FACES[self as usize]
    }

    /// Are these two axes parallel? (+X/-X, +Y/-Y, +Z/-Z, pairs of same-sign axes are also
    /// acceptible +X/+X, -Y/-Y, etc.)
    #[inline]
    pub fn is_parallel(self, other: Axis) -> bool {
        self as u8 % 3 == other as u8 % 3
    }

    /// Get the same axis in the opposite direction. (+X => -X, -Z => +Z)
    #[inline]
    pub fn opposite(self) -> Axis {
        match self {
            Axis::PosX => Axis::NegX,
            Axis::PosY => Axis::NegY,
            Axis::PosZ => Axis::NegZ,
            Axis::NegX => Axis::PosX,
            Axis::NegY => Axis::PosY,
            Axis::NegZ => Axis::PosZ,
        }
    }

    /// Does this axis represent a negative/zero axis?
    ///
    /// Returns true if `self` is `NegX`, `NegY`, or `NegZ`.
    #[inline]
    pub fn is_negative(self) -> bool {
        match self {
            Axis::PosX => false,
            Axis::PosY => false,
            Axis::PosZ => false,
            Axis::NegX => true,
            Axis::NegY => true,
            Axis::NegZ => true,
        }
    }

    /// Does this axis represent a positive/nonzero axis?
    ///
    /// Returns true if `self` is `PosX`, `PosY`, or `PosZ`.
    #[inline]
    pub fn is_positive(self) -> bool {
        !self.is_negative()
    }

    /// Get the centroid of the face on this axis.
    #[inline]
    pub fn to_face_center(self) -> Exact {
        match self {
            Axis::PosX => Exact(Point3::new(2, 1, 1)),
            Axis::PosY => Exact(Point3::new(1, 2, 1)),
            Axis::PosZ => Exact(Point3::new(1, 1, 2)),
            Axis::NegX => Exact(Point3::new(0, 1, 1)),
            Axis::NegY => Exact(Point3::new(1, 0, 1)),
            Axis::NegZ => Exact(Point3::new(1, 1, 0)),
        }
    }

    #[inline]
    pub(crate) fn to_bits(self) -> BitArray<Lsb0, [u8; 1]> {
        BitArray::new([self as u8])
    }

    /// Convenient helper for conversion to `usize`.
    #[inline]
    pub fn to_index(self) -> usize {
        self as u8 as usize
    }

    /// Create an iterator over all six axes.
    #[inline]
    pub fn generator() -> impl Iterator<Item = Self> {
        (0u8..6).map(|b| unsafe { Self::from_u8_unchecked(b) })
    }

    /// True if the face set for this axis requires a winding flip.
    ///
    /// This is likely not useful to you as a user, and is used internally inside the exterior hull
    /// generation algorithm. But it is still useful to document for external understanding:
    ///
    /// # Winding order of exterior faces and face sets
    ///
    /// As a refresher, this is our vertex numbering scheme:
    ///
    /// ```text
    ///   v3      v7
    ///     *----*
    /// v2 /| v6/|
    ///   *----* |   +Y
    ///   | *--|-*   ^ ^ +Z
    ///   |/v1 |/v5  |/
    ///   *----*     +--> +X
    ///  v0    v4
    /// ```
    ///
    /// We want all face windings to be in CCW order. Due to the fact that this order produces a
    /// Z-pattern on each face set - e.g. `0, 1, 2, 3`, `0, 2, 4, 6`, `1, 3, 5, 7`, when sorted -
    /// the exterior facet generation code will flip the first two vertices: `1, 0, 2, 3`, `2, 0, 4,
    /// 6`, etc., producing what should ideally be a CCW-wound quad.
    ///
    /// Unfortunately this is not always the case. Note that the `NegY` axis's face set, when
    /// sorted, produces `0, 1, 4, 5` for its Z-pattern, and `1, 0, 4, 5` for its wound-pattern.
    /// Which, when viewed from below, is a clockwise winding, not counter-clockwise. As such,
    /// unless a better descriptor of *why* stuff must be flipped, we'll need to just straight up
    /// flip `NegY` to compensate.
    ///
    /// Axes that need to be flipped:
    ///
    /// - `NegX`
    /// - `PosY`
    /// - `NegZ`
    ///
    /// Axes that do not need to be flipped:
    ///
    /// - `PosX`
    /// - `NegY`
    /// - `PosZ`
    ///
    /// It seems like the reason that we need things to be flipped differently on the Y axis is that
    /// this ordering is naturally *right-handed*, and our "world" coordinate system is left-handed.
    /// This is just my hypothesis though, and frankly, as is, this flipping is strictly done
    /// because it makes things work.
    #[inline]
    pub fn requires_winding_flip(self) -> bool {
        use Axis::*;
        match self {
            NegX | PosY | NegZ => true,
            PosX | NegY | PosZ => false,
        }
    }
}

/// A set of vertices represented as [`Vertex`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexSet {
    /// The vertices of the set, encoded as a bitset of vertex indices.
    pub bits: BitArray<Lsb0, [u8; 1]>,
}

impl fmt::Display for VertexSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !f.alternate() {
            write!(f, "VertexSet({:?})", self.bits)
        } else {
            write!(f, "VertexSet([")?;
            for (i, v) in self.vertices().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{}", v.to_f32())?;
            }
            write!(f, "])")
        }
    }
}

impl Default for VertexSet {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl VertexSet {
    /// Create an empty vertex set.
    #[inline]
    pub fn new() -> Self {
        Self {
            bits: BitArray::zeroed(),
        }
    }

    /// Create a vertex set from a byte, where each bit represents a set/unset vertex.
    #[inline]
    pub fn from_u8(byte: u8) -> Self {
        Self {
            bits: BitArray::new([byte]),
        }
    }

    /// How many vertices are in this set?
    ///
    /// Always returns a value in the range `0..8`.
    #[inline]
    pub fn len(self) -> usize {
        self.bits.count_ones()
    }

    /// Does this vertex set contain a given vertex?
    #[inline]
    pub fn contains(self, v: Vertex) -> bool {
        *self.bits.get(v.to_u8() as usize).unwrap()
    }

    /// Insert a single vertex into the set.
    ///
    /// You can also use the bitwise "or" operators (`|=` or `|`.)
    #[inline]
    pub fn insert(&mut self, vertex: Vertex) {
        self.bits.set(vertex.to_u8() as usize, true);
    }

    /// The complement of a face set will be its opposing face; this operation is quite useful.
    #[inline]
    pub fn complement(self) -> VertexSet {
        Self { bits: !self.bits }
    }

    /// Test if `self` is a subset of `other`.
    #[inline]
    pub fn subset_of(self, other: VertexSet) -> bool {
        (self & other) == self
    }

    /// Returns `true` if all vertices in this set are coplanar.
    #[inline]
    pub fn is_coplanar(self) -> bool {
        // If we have less than four vertices, then by definition, they are always coplanar.
        // Likewise, we are assured that if we have more than four vertices, we have at least a
        // tetrahedron, so we have a set of vertices which are not entirely coplanar. We are only
        // interested in the case where we could potentially have just a single square face of the
        // cube, that is, when we have exactly four vertices.
        let len = self.len();
        if len != 4 {
            // If we have less than four, we're always coplanar - it's a triangle or less.
            return len < 4;
        }

        // For a set of four vertices to be coplanar, they must be in the set of faces or symmetry
        // planes; we could check against all 12 of these. Alternatively, we can check to see if
        // they contain any of the anomalous configurations. We run with the latter check, in the
        // expectation that the vast majority of sets we test with this check will be non-coplanar
        // (and will give us a chance to exit early out of less total checks anyways.)
        !ANOMALOUS_CONFIGURATIONS.iter().any(|ac| ac.subset_of(self))
    }

    /// Does this vertex set have no set bits?
    #[inline]
    pub fn is_empty(self) -> bool {
        self.bits.not_any()
    }

    /// Iterates over all vertices in the set, in order of their index.
    #[inline]
    pub fn vertices(&self) -> impl Iterator<Item = Vertex> + '_ {
        self.bits.iter_ones().map(|i| Vertex::from_u8(i as u8))
    }

    /// Returns `Some` with the plane of the subset if the given subset is on the convex hull of
    /// this vertex set. Panicks if the subset is degenerate or non-coplanar.
    #[inline]
    pub fn hull_plane(self, subset: VertexSet) -> Option<Plane> {
        assert!(
            subset.len() >= 3 && subset.is_coplanar() && subset.subset_of(self),
            "expected a non-degenerate coplanar subset of this set"
        );

        // 1.) Find the plane on which this subset sits.
        // 2.) Test all the points and see whether or not they sit on the same side of the subset.
        let difference = self - subset;

        // Plane, defined as point and normal.
        let (p0, n) = {
            let mut subset_vs = subset.vertices().take(3).map(Vertex::to_f32);
            let p0 = subset_vs.next().unwrap();
            let p1 = subset_vs.next().unwrap();
            let p2 = subset_vs.next().unwrap();
            let v0 = p1 - p0;
            let v1 = p2 - p0;
            let n = v0.cross(&v1);
            (p0, n)
        };

        let mut difference_vs = difference.vertices().map(Vertex::to_f32);

        let first = match difference_vs.next() {
            // If there are no points which are not in the plane, then it's trivially on the hull
            // (it *is* the hull.)
            None => return Some(Plane { p0, n }),
            Some(v) => v,
        };

        let sign = (first - p0).dot(&n) > 0.;
        // If they all have the same sign against the plane, then this subset is on the convex hull.
        difference_vs
            .all(|p| ((p - p0).dot(&n) > 0.) == sign)
            .then(|| Plane { p0, n })
    }

    /// Returns true if the given subset is on the convex hull of this vertex set. Panicks if the
    /// subset is degenerate or non-coplanar.
    #[inline]
    pub fn is_on_hull(self, subset: VertexSet) -> bool {
        self.hull_plane(subset).is_some()
    }

    /// Convert this vertex set to the representative byte.
    #[inline]
    pub fn to_u8(self) -> u8 {
        self.bits.into_inner()[0]
    }

    /// Create an iterator over all 256 possible vertex sets.
    #[inline]
    pub fn generator() -> impl Iterator<Item = Self> {
        (0..=255u8).map(Self::from_u8)
    }

    /// Find the centroid in `f32` coordinates.
    #[inline]
    pub fn centroid(self) -> Point3<f32> {
        let coords = self
            .vertices()
            .map(Vertex::to_f32)
            .map(|p| p.coords)
            .sum::<Vector3<f32>>()
            / self.len() as f32;

        Point3::from(coords)
    }
}

impl BitOr for VertexSet {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl BitAnd for VertexSet {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits & rhs.bits,
        }
    }
}

impl BitXor for VertexSet {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits ^ rhs.bits,
        }
    }
}

impl Not for VertexSet {
    type Output = Self;
    #[inline]
    fn not(self) -> Self::Output {
        self.complement()
    }
}

impl Sub for VertexSet {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self & !rhs
    }
}

impl BitOrAssign for VertexSet {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitAndAssign for VertexSet {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitXorAssign for VertexSet {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl SubAssign for VertexSet {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl BitOr<Vertex> for VertexSet {
    type Output = VertexSet;
    #[inline]
    fn bitor(mut self, rhs: Vertex) -> Self::Output {
        self.insert(rhs);
        self
    }
}

impl BitAnd<Vertex> for VertexSet {
    type Output = bool;
    #[inline]
    fn bitand(self, rhs: Vertex) -> Self::Output {
        self.contains(rhs)
    }
}

impl BitXor<Vertex> for VertexSet {
    type Output = VertexSet;
    #[inline]
    fn bitxor(self, rhs: Vertex) -> Self::Output {
        self ^ rhs.to_set()
    }
}

impl FromIterator<Vertex> for VertexSet {
    #[inline]
    fn from_iter<T: IntoIterator<Item = Vertex>>(iter: T) -> Self {
        let mut this = VertexSet::new();
        for vertex in iter {
            this.insert(vertex);
        }
        this
    }
}

impl IntoIterator for VertexSet {
    type IntoIter = arrayvec::IntoIter<Vertex, 8>;
    type Item = Vertex;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.vertices().collect::<ArrayVec<Vertex, 8>>().into_iter()
    }
}

/// The error type returned on trying to construct an atom using a degenerate vertex set.
#[derive(Debug, thiserror::Error)]
#[error("cannot construct a valid atom (four or more vertices and enclosing a nonzero volume) from vertex set: {:#}", vertices)]
pub struct DegeneracyError {
    /// The vertices we tried to construct the atom with.
    pub vertices: VertexSet,
}

/// An `Atom` is the smallest individual unit of our collision mesh. It is a convex polyhedron
/// represented as a subset of the vertices of a cube. For an `Atom` to be valid, it must have at
/// least four non-coplanar vertices (it must have nonzero volume.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Atom {
    vertices: VertexSet,
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !f.alternate() {
            write!(f, "Atom({:?})", self.vertices.bits)
        } else {
            write!(f, "Atom([")?;
            for (i, v) in self.vertices.vertices().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{}", v.to_f32())?;
            }
            write!(f, "])")
        }
    }
}

impl Atom {
    /// Construct an atom from a vertex set. Panicks if the vertex set is coplanar/has zero volume.
    #[inline]
    pub fn new(vertices: VertexSet) -> Self {
        Self::try_new(vertices).unwrap()
    }

    /// Try to construct an atom from a given vertex set.
    #[inline]
    pub fn try_new(vertices: VertexSet) -> Result<Self, DegeneracyError> {
        if vertices.is_coplanar() {
            Err(DegeneracyError { vertices })
        } else {
            Ok(Self { vertices })
        }
    }

    /// Convert this atom back to its internal vertex set.
    #[inline]
    pub fn to_set(self) -> VertexSet {
        self.vertices
    }

    /// Compute the compound hull of this atom (the combined interior and exterior hulls, in a
    /// representation which allows for fast and easy clipping.)
    #[inline]
    pub fn compound_hull(self) -> CompoundHull {
        CompoundHull::new(self)
    }

    /// Convert this atom to the byte representation of its vertex set.
    #[inline]
    pub fn to_u8(self) -> u8 {
        self.vertices.to_u8()
    }

    /// Create an iterator over all possible valid atoms.
    #[inline]
    pub fn generator() -> impl Iterator<Item = Self> {
        VertexSet::generator().filter_map(|vs| Self::try_new(vs).ok())
    }
}

/// A point in "exact" cube coordinates; integer coordinates scaled by a factor of two.
///
/// This means that the unit cube in "exact" coordinates is really the cube of side length two. We
/// do this so that we can represent facets which have vertices necessarily in the center of a side.
/// This allows us to exactly represent all the vertices of a mesh of joined atoms, in a way which
/// can be conveniently hashed and exactly compared in order to deduplicate vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct Exact(pub Point3<i32>);

impl Exact {
    /// Convert this exact vertex to a `Point3<f32>`.
    pub fn to_f32(self) -> Point3<f32> {
        self.0.cast::<f32>()
    }
}
