use std::{collections::HashMap, fmt, ops::Deref};

use arrayvec::ArrayVec;
use bitvec::prelude::*;
use nalgebra::*;
use slab::Slab;

use crate::{Atom, Axis, Exact, Vertex, VertexSet, ANOMALOUS_CONFIGURATIONS, SYMMETRIES};

/// A `Face` represents a combination of four triangles around a central vertex:
///
/// *---*
/// |\2/|
/// |3*1|
/// |/0\|
/// * - *
///
/// Where a * is a vertex and anything else is empty space. These four triangles make up the bulk of
/// how we mesh two atoms together: combining two atoms' adjacent faces is as simple as XORing the
/// bits of their faces together. We immediately zero out any shared triangles.
///
/// A `Face` will only be non-empty if its corresponding atom has more than two vertices on its
/// plane; its bits represent strictly filled spaces, not lines. a `Face` which does not actually
/// have any triangles on its surface will not contain any set bits.
///
/// Faces are constructed in a right-handed coordinate system according to the axis of the face of
/// the atom from which they were constructed; this means that for example, if we're looking at the
/// +X axis of an atom, then to the "right" ([`FACE_POS_X`]) will be `-Z` (atom vertices with
/// unset [`Z_AXIS`] bits.)
///
/// *Parallel faces share the same layout with respect to their other axes.* For example, a face on
/// the `-Z` axis will have all of its vertices parallel w/ a face on the `Z` axis; the vertices are
/// stored in the same order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Face {
    bits: BitArray<Lsb0, [u8; 1]>,
}

impl Face {
    /// Calculate an initial `Face` object from a vertex set and an axis corresponding to the face
    /// we want to construct on.
    #[inline]
    pub fn new(vertices: VertexSet, axis: Axis) -> Self {
        let face_set = axis.to_face_set();
        let mut face_verts = BitArray::<Lsb0, [u8; 1]>::zeroed();
        for (i, v) in face_set.vertices().enumerate() {
            face_verts.set(i, vertices.contains(v));
        }

        let mut bits = BitArray::zeroed();
        // We either have exactly all of the face triangles, or half of them, or none of them.
        if face_verts.count_ones() == 4 {
            bits.set_all(true);
        } else if face_verts.count_ones() == 3 {
            // All of our sides follow the same Z pattern, which quite literally looks like a Z. So,
            // we can use the same logic in all cases. The Z starts at the bottom right, with vertex
            // 0; then goes to vertex 1, vertex 2, and vertex 3. We will number these vertices in
            // the way described in the `Face` documentation:
            // v3   v2
            //  *---*
            //  |\2/|
            //  |3*1|
            //  |/0\|
            //  * - *
            // v1   v0
            //
            // Given that, for any unset vertex, we know we have the right triangle composed of all
            // the other vertices - which encloses exactly two of the four face triangles:
            // - Vertex 0 unset - triangles formed by v123 (triangles 2 and 3)
            // - Vertex 1 unset - triangles formed by v023 (triangles 1 and 2)
            // - Vertex 2 unset - triangles formed by v013 (triangles 0 and 3)
            // - Vertex 3 unset - triangles formed by v012 (triangles 0 and 1)

            // Which bit is *not* set?
            let unset = face_verts.first_zero().unwrap();
            match unset {
                0 => bits[0..4].copy_from_bitslice(bits![Lsb0, u8; 0, 0, 1, 1]),
                1 => bits[0..4].copy_from_bitslice(bits![Lsb0, u8; 0, 1, 1, 0]),
                2 => bits[0..4].copy_from_bitslice(bits![Lsb0, u8; 1, 0, 0, 1]),
                3 => bits[0..4].copy_from_bitslice(bits![Lsb0, u8; 1, 1, 0, 0]),
                _ => unreachable!(),
            }
        }

        bits[4..7].copy_from_bitslice(&axis.to_bits()[0..3]);

        Self { bits }
    }

    /// Construct an empty face on a given axis.
    ///
    /// Atoms can't be empty, so this is useful if you need to just conjure an empty face out of
    /// nowhere.
    #[inline]
    pub fn empty(axis: Axis) -> Self {
        let mut bits = BitArray::zeroed();
        bits[4..7].copy_from_bitslice(&axis.to_bits()[0..3]);

        Self { bits }
    }

    /// The axis which this face was constructed for.
    #[inline]
    pub fn axis(self) -> Axis {
        let mut axis_bits = BitArray::zeroed();
        axis_bits[0..3].copy_from_bitslice(&self.bits[4..7]);
        unsafe { Axis::from_u8_unchecked(axis_bits.into_inner()) }
    }

    /// Join two opposing faces. This mutually removes any triangles which are shared between the
    /// faces. Panicks if called on faces which are not made from opposing axes (where opposing axes
    /// are pairs like `PosX` vs `NegX`, etc.)
    #[inline]
    pub fn join(&mut self, other: &mut Self) {
        assert!(
            self.axis() == other.axis().opposite(),
            "cannot join faces with non-opposing axes!"
        );

        let mut xord = self.bits;
        xord[0..4] ^= other.bits[0..4].iter().by_val();

        self.bits[0..4] &= xord[0..4].iter().by_val();
        other.bits[0..4] &= xord[0..4].iter().by_val();
    }

    /// Iterate over the exactly one or two facets of this face.
    #[inline]
    pub fn facets(self) -> impl Iterator<Item = HullFacet> {
        // There are sixteen possible configurations of the facets of a `Face`. Let's count:
        // - (1) No facet bits (no triangle :C)
        // - (4) One facet bit (one smol triangle)
        // - (6) Two facet bits (two possible classes of subconfigurations)
        //     - (2) Two adjacent facet bits (one beeg triangle)
        //     - (4) Two non-adjacent facet bits (two smol triangle)
        // - (4) Three facet bits (one beeg triangle, one smol triangle)
        // - (1) All facet bits (one square)
        //
        // We handle these via a brute-force lookup, as there are only sixteen of them.

        // Mask off the index to get our first four bits.
        let index = self.bits.into_inner()[0] & 0b1111;
        let axis = self.axis();
        let mut face_verts: ArrayVec<Vertex, 4> = axis.to_face_set().into_iter().collect();
        // Swap the 0th and 1st vertices to get the vertices to match our Z pattern. This now means
        // that if we have some facet index `f`, the vertices that make up its edge are now
        // `face_verts[f]`, `face_verts[f + 1]` (with "wrapping" behavior at the end of the array.)
        face_verts.swap(0, 1);
        let center = axis.to_face_center();

        let square = || {
            let mut verts = face_verts
                .clone()
                .into_inner()
                .unwrap()
                .map(Vertex::to_exact);

            // Winding order.
            if self.axis().requires_winding_flip() {
                verts.swap(1, 3);
            }

            HullFacet::Rectangle(verts)
        };

        // beeg triangle: from a run of two adjacent facets
        let beeg = |start_facet: usize| {
            let v1 = start_facet;
            let v2 = (v1 + 1) & 0b11;
            let v3 = (v2 + 1) & 0b11;

            // flip winding order
            if self.axis().requires_winding_flip() {
                HullFacet::Triangle(
                    [face_verts[v2], face_verts[v1], face_verts[v3]].map(Vertex::to_exact),
                )
            } else {
                HullFacet::Triangle(
                    [face_verts[v1], face_verts[v2], face_verts[v3]].map(Vertex::to_exact),
                )
            }
        };

        let smol = |facet: usize| {
            let v1 = facet;
            let v2 = (v1 + 1) & 0b11;

            // flip winding order.
            if self.axis().requires_winding_flip() {
                HullFacet::Triangle([center, face_verts[v2].to_exact(), face_verts[v1].to_exact()])
            } else {
                HullFacet::Triangle([center, face_verts[v1].to_exact(), face_verts[v2].to_exact()])
            }
        };

        let av = |slice: &[HullFacet]| slice.iter().copied().collect::<ArrayVec<HullFacet, 2>>();

        let facets = match index {
            // (0) No facet bits.
            0b0000 => av(&[]),
            // (1) One facet bit (one smol triangle).
            0b0001 => av(&[smol(0)]),
            // (2) One facet bit (one smol triangle).
            0b0010 => av(&[smol(1)]),
            // (3) Two adjacent facet bits (one beeg triangle).
            0b0011 => av(&[beeg(0)]),
            // (4) One facet bit (one smol triangle).
            0b0100 => av(&[smol(2)]),
            // (5) Two non-adjacent facet bits (two smol triangle).
            0b0101 => av(&[smol(0), smol(2)]),
            // (6) Two adjacent facet bits (one beeg triangle).
            0b0110 => av(&[beeg(1)]),
            // (7) Three facet bits (one beeg one smol).
            0b0111 => av(&[beeg(0), smol(2)]),
            // (8) One facet bit (one smol triangle).
            0b1000 => av(&[smol(3)]),
            // (9) Two adjacent facet bits (one beeg triangle).
            0b1001 => av(&[beeg(3)]),
            // (10) Two non-adjacent facet bits (two smol triangle).
            0b1010 => av(&[smol(1), smol(3)]),
            // (11) Three facet bits (one beeg one smol).
            0b1011 => av(&[beeg(0), smol(3)]),
            // (12) Two adjacent facet bits (one beeg triangle).
            0b1100 => av(&[beeg(2)]),
            // (13) Three facet bits (one beeg one smol).
            0b1101 => av(&[beeg(2), smol(0)]),
            // (14) Three facet bits (one beeg one smol).
            0b1110 => av(&[beeg(1), smol(3)]),
            // (15) All facet bits (one square).
            0b1111 => av(&[square()]),

            // Not a 4-bit index.
            _ => unreachable!(),
        };

        facets.into_iter()
    }
}

/// A set of exactly two distinct vertices, representing an edge of an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    two_bits: VertexSet,
}

impl Edge {
    /// Create a new edge from a pair of vertices.
    pub fn new(v1: Vertex, v2: Vertex) -> Self {
        assert_ne!(v1, v2, "edges must consist of two distinct vertices!");
        Self {
            two_bits: [v1, v2].into_iter().collect(),
        }
    }

    /// Convert this pair of vertices to a [`VertexSet`].
    pub fn to_set(self) -> VertexSet {
        self.two_bits
    }

    /// Iterate over all possible edges of the unit cube (12 in total.)
    pub fn generator() -> impl Iterator<Item = Edge> {
        let vs: ArrayVec<Vertex, 8> = Vertex::generator().collect();
        [
            Self::new(vs[0], vs[1]),
            Self::new(vs[1], vs[3]),
            Self::new(vs[3], vs[2]),
            Self::new(vs[2], vs[0]),
            Self::new(vs[4], vs[5]),
            Self::new(vs[5], vs[7]),
            Self::new(vs[7], vs[6]),
            Self::new(vs[6], vs[4]),
            Self::new(vs[0], vs[4]),
            Self::new(vs[1], vs[5]),
            Self::new(vs[2], vs[6]),
            Self::new(vs[3], vs[7]),
        ]
        .into_iter()
    }
}

/// Triangles which lie on the planes of the faces of an atom are taken care of by `Face`s. However,
/// we still need to complete the picture by adding back in any missing facets which lie "inside"
/// that cube (and not on its faces.) There are 8 ways to choose 3 triangles from eight
/// vertices. Of these 56 possible triangles, there are 4 times 24 which exist on the faces of the
/// cube, and which are therefore superseded by the `Face` calculations. We now have 32 triangles
/// remaining. Of these, we have six "symmetry planes" of the cube; these are the planes that run
/// through an edge each, and both contain four vertices. Like the face quads, these contain 24
/// possible triangles, which leaves us with a total of 8 anomalous configurations. To sum up (pun
/// intended):
///
/// - 56 total triangles (8 choose 3). (Given a quad, there are four triangles in it; divide it in
///   half, then flip the edge to get the other two.)
///     - 6 quads in cube faces = 24 triangle configurations
///     - 6 quads in symmetry planes = 24 triangle configurations
///     - 8 remaining triangles (anomalous configurations)
///
/// 32 of these configurations are ones we're interested in. 24 of these can be enumerated by
/// enumerating the symmetry planes, and at the same time can be collapsed into a quad in order to
/// remain as simplified as possible (or we can triangulate them for good measure.) The remaining 8
/// are more troublesome, but can be enumerated as sets where no vertex shares an edge. How can we
/// know this? Let's think through it.
///
/// Suppose a vertex shares an edge with another vertex in this triangle. There are three other
/// edges on which to choose the last vertex. Two of them would put the triangle on the faces of the
/// cube. The remaining option will place the triangle on one of the symmetry planes.
///
/// So, it seems obvious that the remaining configurations are those where the vertices don't even
/// share an edge, let alone a face. The number of these is only eight, so we can enumerate them
/// (see [`ANOMALOUS_CONFIGURATIONS`].) These are simple to deal with as well, as they can never be
/// collapsed into a quad, at least not within a single `Atom`, because these are the groups of
/// three vertices which are alone on their plane within the cube.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct InteriorHull {
    facets: ArrayVec<HullFacet, 4>,
}

impl InteriorHull {
    /// Generate the interior hull of an atom.
    #[inline]
    pub fn new(atom: Atom) -> Self {
        generate_interior_hull(atom.vertices)
    }

    /// Return an iterator over all facets on the interior hull.
    #[inline]
    pub fn facets(&self) -> impl Iterator<Item = HullFacet> {
        self.facets.clone().into_iter()
    }
}

#[inline]
fn generate_interior_hull(atom: VertexSet) -> InteriorHull {
    let mut facets = ArrayVec::new();
    let centroid = atom.centroid();

    // First, check all of the symmetry planes; extract all vertices of each symmetry plane. If
    // there are more than two vertices, we found a potentially on-hull intersection. Try it as a
    // square; if it doesn't work, break it into triangles and see if we can make any of the pieces
    // fit.
    //
    // Once we're done with the symmetry planes, swap to the anomalous configurations.
    //
    // This is probably actually fast enough to do in real-time?
    for intersection in SYMMETRIES
        .into_iter()
        .chain(ANOMALOUS_CONFIGURATIONS)
        .map(|configuration| configuration & atom)
        .filter(|intersection| intersection.len() >= 3)
    {
        if intersection.len() > 2 && atom.is_on_hull(intersection) {
            let mut facet = HullFacet::from_vertex_set(intersection);
            facet.match_winding_to_centroid(&centroid);
            facets.push(facet);
        } else if intersection.len() == 4 {
            // Permute the triangles and try them all.
            let mut vs = intersection.vertices();
            let tris: [Vertex; 4] = [
                vs.next().unwrap(),
                vs.next().unwrap(),
                vs.next().unwrap(),
                vs.next().unwrap(),
            ];

            // Try:
            // - 0, 1, 2
            // - 0, 1, 3
            // - 0, 2, 3
            // - 1, 2, 3
            //
            // Since we just ruled out the combination of both, only one could be on the hull.

            let tri_lists = [
                [tris[0], tris[1], tris[2]],
                [tris[0], tris[1], tris[3]],
                [tris[0], tris[2], tris[3]],
                [tris[1], tris[2], tris[3]],
            ];

            for tri_list in tri_lists {
                let tri_set = tri_list.into_iter().collect();
                if atom.is_on_hull(tri_set) {
                    let mut facet = HullFacet::from_vertex_set(tri_set);
                    facet.match_winding_to_centroid(&centroid);
                    facets.push(facet);
                    // Only one can be on the hull!
                    break;
                }
            }
        }
    }

    InteriorHull { facets }
}

/// Facets which are on an atom's convex hull and which exist on the six "face planes" of the cube,
/// and which may be clipped during `join` operations.
///
/// Internally, this is a set of six [`Face`]s.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExteriorHull {
    faces: ArrayVec<Face, 6>,
}

impl ExteriorHull {
    /// Calculate the initial exterior hull of an atom (the exterior hull before any facets are
    /// clipped out due to join operations.)
    #[inline]
    pub fn new(atom: Atom) -> Self {
        let faces = Axis::generator()
            .map(|axis| Face::new(atom.to_set(), axis))
            .collect();
        Self { faces }
    }

    /// Join two exterior hulls, along a given axis. The axis is assumed to be relative to `self`,
    /// with the chosen face to join on the other hull given by the opposite (negated) axis.
    #[inline]
    pub fn join(&mut self, axis: Axis, other: &mut ExteriorHull) {
        let self_axis = axis;
        let other_axis = axis.opposite();
        let self_face = &mut self.faces[self_axis.to_index()];
        let other_face = &mut other.faces[other_axis.to_index()];
        self_face.join(other_face);
    }

    /// Iterate over all remaining facets of the exterior hull.
    #[inline]
    pub fn facets(&self) -> impl Iterator<Item = HullFacet> + '_ {
        self.faces.iter().copied().flat_map(Face::facets)
    }

    /// Directly set a face of the exterior hull.
    #[inline]
    pub fn set_face(&mut self, face: Face) {
        self.faces[face.axis().to_index()] = face;
    }

    /// Get a face of the exterior hull.
    #[inline]
    pub fn face(&self, axis: Axis) -> &Face {
        &self.faces[axis.to_index()]
    }
}

/// The "compound hull" of an atom is its convex hull, possibly with pieces missing from its
/// exterior.
///
/// It comprises the union of two sets:
/// - The "interior hull" ([`InteriorHull`]), which comprises the convex hull of the atom with all
///   facets on the faces of the unit cube removed.
/// - The "exterior hull" ([`ExteriorHull`]), which begins by comprising the convex hull of the atom
///   consisting only of faces on the faces of the unit cube, and which may have facets subtracted
///   from it during `join` operations.
///
/// In the end, the interior hull of an atom should never change; however, the exterior hull may
/// change depending on adjacent atoms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct CompoundHull {
    interior: InteriorHull,
    exterior: ExteriorHull,
    edges: ArrayVec<Edge, 12>,
}

impl CompoundHull {
    /// Calculate the initial compound hull (before join operations) of an atom.
    #[inline]
    pub fn new(atom: Atom) -> Self {
        Self {
            interior: InteriorHull::new(atom),
            exterior: ExteriorHull::new(atom),
            edges: Edge::generator()
                .filter(|&edge| edge.to_set().subset_of(atom.to_set()))
                .collect(),
        }
    }

    /// Get the interior component of the hull.
    #[inline]
    pub fn interior(&self) -> &InteriorHull {
        &self.interior
    }

    /// Get the exterior component of the hull.
    #[inline]
    pub fn exterior(&self) -> &ExteriorHull {
        &self.exterior
    }

    /// Get the exterior component of the hull, mutably.
    #[inline]
    pub fn exterior_mut(&mut self) -> &mut ExteriorHull {
        &mut self.exterior
    }

    /// Join the exterior hulls of two compound hulls.
    ///
    /// See [`ExteriorHull::join`].
    #[inline]
    pub fn join_exteriors(&mut self, axis: Axis, other: &mut Self) {
        self.exterior.join(axis, &mut other.exterior)
    }

    /// Iterate over all facets on this hull.
    #[inline]
    pub fn facets(&self) -> impl Iterator<Item = HullFacet> + '_ {
        self.interior.facets().chain(self.exterior.facets())
    }
}

/// A facet of a hull is either a rectangle or a triangle.
///
/// Facets should follow a CCW winding order for both rectangles and triangles. Compound hulls will
/// never produce a set of facets from a single atom which can be simplified by joining triangles
/// into squares.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum HullFacet {
    /// A hull facet which is a triangle.
    Triangle([Exact; 3]),
    /// A hull facet which is a rectangle.
    Rectangle([Exact; 4]),
}

impl fmt::Debug for HullFacet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Triangle([a, b, c]) => write!(f, "Triangle([{}, {}, {}])", a.0, b.0, c.0),
            Self::Rectangle([a, b, c, d]) => {
                write!(f, "Rectangle([{}, {}, {}, {}])", a.0, b.0, c.0, d.0)
            }
        }
    }
}

impl HullFacet {
    /// Translate every point in the facet by the given vector *in atom coordinates* (not exact
    /// coordinates.)
    #[inline]
    pub fn translated_by(self, v: Vector3<i32>) -> Self {
        match self {
            Self::Triangle([a, b, c]) => Self::Triangle([a + v, b + v, c + v]),
            Self::Rectangle([a, b, c, d]) => Self::Rectangle([a + v, b + v, c + v, d + v]),
        }
    }

    /// Find the facet's normal.
    #[inline]
    pub fn normal(self) -> UnitVector3<f32> {
        UnitVector3::new_normalize(self.discrete_scaled_normal().cast::<f32>())
    }

    /// Calculated the scaled normal of the face (the un-normalized normal - yes, I know, silly
    /// vocabulary) in such a way that it is exactly comparable.
    ///
    /// The actual scaling involved is undefined, but you can expect it is the same for any facet
    /// from this library (so if facets were calculated from coplanar sets of vertices they will be
    /// identical.)
    #[inline]
    pub fn discrete_scaled_normal(self) -> Vector3<i32> {
        let (Self::Triangle([a, b, c]) | Self::Rectangle([a, b, c, _])) = self;
        let (p0, p1, p2) = (a.0, b.0, c.0);
        let v01 = p1 - p0;
        let v02 = p2 - p0;
        v01.cross(&v02)
    }

    /// Test if this facet's normal faces outwards from a given point, such as the centroid of the
    /// vertex set. Useful for asserting that a facet is properly wound CCW.
    #[inline]
    pub fn is_normal_outwards_with_respect_to_point(self, p0: &Point3<f32>) -> bool {
        let (Self::Triangle([a, _, _]) | Self::Rectangle([a, _, _, _])) = self;
        let p1 = a.to_f32();
        let n = self.normal();
        let v = p1 - p0;
        n.dot(&v) > 0.
    }

    /// Given a centroid, ensure that the winding order of this facet produces a normal that faces
    /// outwards from it.
    #[inline]
    pub fn match_winding_to_centroid(&mut self, centroid: &Point3<f32>) {
        if !self.is_normal_outwards_with_respect_to_point(centroid) {
            self.flip_winding();
        }
    }

    /// Flip the vertex winding of this hull facet.
    #[inline]
    pub fn flip_winding(&mut self) {
        match self {
            Self::Triangle([a, b, _]) => std::mem::swap(a, b),
            Self::Rectangle([_, b, _, d]) => std::mem::swap(b, d),
        }
    }

    /// Construct a new hull facet from a vertex set.
    #[inline]
    pub fn from_vertex_set(vs: VertexSet) -> Self {
        match vs.len() {
            3 => HullFacet::Triangle(
                vs.vertices()
                    .map(Vertex::to_exact)
                    .collect::<ArrayVec<_, 3>>()
                    .into_inner()
                    .unwrap(),
            ),
            4 => {
                let mut array = vs
                    .vertices()
                    .map(Vertex::to_exact)
                    .collect::<ArrayVec<_, 4>>()
                    .into_inner()
                    .unwrap();
                // Fix "Z" pattern.
                array.swap(0, 1);
                HullFacet::Rectangle(array)
            }
            _ => unreachable!("all facets must have 3 or 4 vertices!"),
        }
    }

    #[inline]
    fn exterior_edges(self) -> impl Iterator<Item = SortedPair<Exact>> {
        match self {
            HullFacet::Triangle([a, b, c]) => [
                SortedPair::new(a, b),
                SortedPair::new(b, c),
                SortedPair::new(c, a),
            ]
            .into_iter()
            .collect::<ArrayVec<SortedPair<Exact>, 6>>(),
            HullFacet::Rectangle([a, b, c, d]) => [
                SortedPair::new(a, b),
                SortedPair::new(b, c),
                SortedPair::new(a, c),
                SortedPair::new(c, d),
                SortedPair::new(d, a),
                SortedPair::new(c, a),
            ]
            .into_iter()
            .collect::<ArrayVec<SortedPair<Exact>, 6>>(),
        }
        .into_iter()
    }

    // #[inline]
    // pub fn interior_edges(self) -> impl Iterator<Item = SortedPair<Exact>> {
    //     match self {
    //         HullFacet::Triangle(_) => ArrayVec::new().into_iter(),
    //         HullFacet::Rectangle([a, b, c, d]) => {
    //             ArrayVec::from([SortedPair::new(a, c), SortedPair::new(b, d)]).into_iter()
    //         }
    //     }
    // }
}

/// A sorted pair of values. Useful to represent an unordered pair as a key in a hashmap/ordered
/// map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SortedPair<T> {
    tuple: (T, T),
}

impl<T: Ord> SortedPair<T> {
    /// Create a new sorted pair, swapping the order of the two elements if necessary to preserve
    /// the ordering property.
    pub fn new(a: T, b: T) -> Self {
        if a < b {
            Self { tuple: (a, b) }
        } else {
            Self { tuple: (b, a) }
        }
    }
}

impl<T> Deref for SortedPair<T> {
    type Target = (T, T);
    fn deref(&self) -> &Self::Target {
        &self.tuple
    }
}

#[derive(Debug)]
enum EdgeStatus {
    Extant(ArrayVec<usize, 2>),
    Removed,
}

impl Default for EdgeStatus {
    fn default() -> Self {
        Self::Extant(ArrayVec::new())
    }
}

impl EdgeStatus {
    fn is_removed(&self) -> bool {
        matches!(self, Self::Removed)
    }
}

#[derive(Debug, Clone, Copy)]
struct FilteredFacet {
    facet: HullFacet,
    /// The discrete scaled normal of the facet.
    dsn: Vector3<i32>,
    atom_coords: Vector3<i32>,
}

impl FilteredFacet {
    fn new(atom_coords: Vector3<i32>, facet: HullFacet) -> Self {
        Self {
            facet,
            dsn: facet.discrete_scaled_normal(),
            atom_coords,
        }
    }

    fn find_perpendicular_axis(&self, other: &FilteredFacet) -> Vector3<i32> {
        let tmp = other.atom_coords - self.atom_coords;
        let parallel_axis = self.dsn.cross(&tmp);
        let perpendicular_axis = parallel_axis.cross(&self.dsn);
        perpendicular_axis
    }
}

/// A data structure for building a set of edges which are considered "interior" and irrelevant to
/// collision normals.
///
/// Any collision with a removed edge should be considered to instead be a collision with the normal
/// of the facet the collision occurred on.
#[derive(Debug, Default)]
pub struct EdgeFilter {
    facet_ids: HashMap<HullFacet, usize>,
    edges: HashMap<SortedPair<Exact>, EdgeStatus>,
    facets: Slab<FilteredFacet>,
}

impl EdgeFilter {
    /// Create an empty edge filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the edge filter, retaining allocated memory for reuse.
    pub fn clear(&mut self) {
        self.facet_ids.clear();
        self.edges.clear();
        self.facets.clear();
    }

    /// Returns true if a facet containing the edge has not been removed/made redundant.
    pub fn edge_exists(&self, edge: SortedPair<Exact>) -> bool {
        self.edges
            .get(&edge)
            .map_or(true, |status| !status.is_removed())
    }

    /// Push all edges of a facet into the filter. This method requires the coordinates of the atom
    /// the facet is coming from as well, as it may be used to calculate a basis to decide whether
    /// two faces are at a less or greater than parallel angle.
    pub fn push(&mut self, atom_coords: Vector3<i32>, facet: HullFacet) {
        use std::collections::hash_map::Entry;

        if let Entry::Vacant(vacant) = self.facet_ids.entry(facet) {
            let filtered_facet = FilteredFacet::new(atom_coords, facet);
            let facet_id = self.facets.insert(filtered_facet);
            vacant.insert(facet_id);

            // println!(
            //     "Fresh facet (id {}) with dsn {:?}: {:?}",
            //     facet_id, facet_dsn, facet
            // );

            'register_edges: for edge in filtered_facet.facet.exterior_edges() {
                let entry = self.edges.entry(edge).or_default();
                let associated_facets = match entry {
                    EdgeStatus::Extant(associated) => associated,
                    EdgeStatus::Removed => continue,
                };

                // This loop exists as a sort of pseudo-goto in order to get around some
                // borrowchecking stuff. Breaking from it interrupts the borrow of
                // `associated_facets`. It will only ever hit the setting of the edge status to
                // `Removed` if it doesn't naturally finish the for loop inside.
                'remove: loop {
                    for &associated_facet_id in associated_facets.iter() {
                        let associated_facet = &self.facets[associated_facet_id];
                        let same_atom = filtered_facet.atom_coords == associated_facet.atom_coords;
                        // let mut debug_perp_axis = None;
                        let filtered_out = if same_atom {
                            // If these two facets come from the same atom, then the perp axis trick
                            // won't work, because the difference in their coordinates is zero. But,
                            // since they're from the same atom, we can just directly compare their
                            // normals.
                            filtered_facet.dsn == associated_facet.dsn
                        } else {
                            // If they *don't* come from the same atom, we can find a vector which
                            // is parallel to the first facet, perpendicular to the edge and the
                            // first facet's normal, and points towards the second facet.
                            // Projecting the second facet's normal on that vector will tell us
                            // whether the angle is less than or equal to 180 degrees (<= 0) or
                            // greater than 180 degrees (> 0.)
                            let perp_axis =
                                filtered_facet.find_perpendicular_axis(associated_facet);
                            // debug_perp_axis = Some(perp_axis);
                            perp_axis.dot(&associated_facet.dsn) <= 0
                        };

                        // println!(
                        //     "{} edge {} <=> {} due to facets ({}, {}) and ({}, {})\n\t\
                        //             perp_axis: {}, dsn: {}, associated_dsn: {}",
                        //     if !filtered_out { "KEEP" } else { "REMOVE" },
                        //     edge.0 .0,
                        //     edge.1 .0,
                        //     Point3::from(atom_coords),
                        //     facet_id,
                        //     Point3::from(associated_facet.atom_coords),
                        //     associated_facet_id,
                        //     match debug_perp_axis {
                        //         Some(axis) => format!("{}", Point3::from(axis)),
                        //         None => "None".to_owned(),
                        //     },
                        //     Point3::from(filtered_facet.dsn),
                        //     Point3::from(associated_facet.dsn),
                        // );

                        // If we have another facet which is attached to this same edge and which
                        // forms a <= 180 degree angle with this this facet, remove the edge.
                        if filtered_out {
                            break 'remove;
                        }

                        // // If we have another facet which is attached to this same edge and which
                        // // shares a plane w/ this facet, then this edge is to be filtered out.
                        // if associated_facet.dsn == facet.dsn
                        //     || -associated_facet.dsn
                        //         == facet.dsn
                        // {
                        //     // println!(
                        //     //     "edge {:?} due to facets {} and {}",
                        //     //     edge, facet_id, associated_facet
                        //     // );
                        //     break 'remove;
                        // }
                    }

                    associated_facets.push(facet_id);
                    continue 'register_edges;
                }

                *entry = EdgeStatus::Removed;
            }
        }
    }

    /// Returns pairs `(is_removed, edge)`.
    pub fn iter(&self) -> impl Iterator<Item = (bool, SortedPair<Exact>)> + '_ {
        self.edges
            .iter()
            .map(|(&e, status)| (status.is_removed(), e))
    }

    /// Returns an iterator over all extant (non-removed) edges.
    pub fn iter_extant(&self) -> impl Iterator<Item = SortedPair<Exact>> + '_ {
        self.edges
            .iter()
            .filter(|(_, status)| !status.is_removed())
            .map(|(&e, _)| e)
    }

    /// Returns an iterator over all removed edges.
    pub fn iter_removed(&self) -> impl Iterator<Item = SortedPair<Exact>> + '_ {
        self.edges
            .iter()
            .filter(|(_, status)| status.is_removed())
            .map(|(&e, _)| e)
    }
}

impl Extend<(Vector3<i32>, HullFacet)> for EdgeFilter {
    fn extend<T: IntoIterator<Item = (Vector3<i32>, HullFacet)>>(&mut self, iter: T) {
        for (atom_coord, facet) in iter {
            self.push(atom_coord, facet);
        }
    }
}

#[derive(Debug)]
struct VertexStatus {
    associated_edges: ArrayVec<usize, 8>,
    removed: bool,
}

impl Default for VertexStatus {
    fn default() -> Self {
        Self {
            associated_edges: ArrayVec::new(),
            removed: false,
        }
    }
}

/// A data structure for building a set of vertices which are considered "interior" and are not
/// relevant to collision normals.
///
/// If a collision occurs and hits one of these vertices, the collision is still valid, but the
/// normal should be adjusted and replaced with the normal of the face being collided with.
///
/// TODO: is the face normal the correct normal for this or should it actually be replaced with the
/// next nearest *edge* being collided with?
#[derive(Debug, Default)]
pub struct VertexFilter {
    edge_ids: HashMap<SortedPair<Exact>, usize>,
    vertices: HashMap<Exact, VertexStatus>,
    edges: Slab<SortedPair<Exact>>,
}

impl VertexFilter {
    /// Create an empty vertex filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the vertex filter, retaining allocated memory for reuse.
    pub fn clear(&mut self) {
        self.edge_ids.clear();
        self.vertices.clear();
        self.edges.clear();
    }

    /// Returns true if a facet containing the edge exists in the edge filter *and* the edge has not
    /// been removed/made redundant.
    pub fn vertex_exists(&self, vertex: Exact) -> bool {
        self.vertices
            .get(&vertex)
            .map_or(false, |status| !status.removed)
    }

    /// Returns associated edges of a vertex.
    pub fn associated_edges(&self, vertex: Exact) -> impl Iterator<Item = &SortedPair<Exact>> {
        let edge_ids = self
            .vertices
            .get(&vertex)
            .map_or(ArrayVec::new(), |status| status.associated_edges.clone());
        edge_ids.into_iter().map(|edge_id| &self.edges[edge_id])
    }

    /// Push a potentially filtered-out edge into the vertex filter.
    pub fn push(&mut self, removed: bool, edge: SortedPair<Exact>) {
        use std::collections::hash_map::Entry;

        if let Entry::Vacant(vacant) = self.edge_ids.entry(edge) {
            let edge_id = self.edges.insert(edge);
            vacant.insert(edge_id);
            let edge_dir = edge.1 .0 - edge.0 .0;

            for vertex in [edge.0, edge.1] {
                let entry = self.vertices.entry(vertex).or_default();

                for &associated_edge_id in entry.associated_edges.iter() {
                    let associated_edge = &self.edges[associated_edge_id];
                    // If we have another edge which is attached to this same vertex and which
                    // shares a direction w/ this edge, then this vertex is to be filtered out.
                    let associated_dir = associated_edge.1 .0 - associated_edge.0 .0;
                    if associated_dir == edge_dir || -associated_dir == edge_dir {
                        entry.removed = true;
                    }
                }

                if !removed {
                    entry.associated_edges.push(edge_id);
                } else {
                    entry.removed = true;
                }
            }
        }
    }
}

impl Extend<(bool, SortedPair<Exact>)> for VertexFilter {
    fn extend<T: IntoIterator<Item = (bool, SortedPair<Exact>)>>(&mut self, iter: T) {
        for (removed, edge) in iter {
            self.push(removed, edge);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{vertex_set, FACES};
    use maplit::hashset;
    use std::collections::HashSet;

    fn hull(vertex_set: VertexSet) -> CompoundHull {
        Atom::new(vertex_set).compound_hull()
    }

    macro_rules! bad_atoms {
        ($($name:ident, $set:expr;)*) => {$(
            #[test]
            #[should_panic]
            fn $name() {
                Atom::new($set);
            }
        )*};
    }

    bad_atoms! {
        bad_atom_f0, FACES[0];
        bad_atom_f1, FACES[1];
        bad_atom_f2, FACES[2];
        bad_atom_f3, FACES[3];
        bad_atom_f4, FACES[4];
        bad_atom_f5, FACES[5];
        bad_atom_s0, SYMMETRIES[0];
        bad_atom_s1, SYMMETRIES[1];
        bad_atom_s2, SYMMETRIES[2];
        bad_atom_s3, SYMMETRIES[3];
        bad_atom_s4, SYMMETRIES[4];
        bad_atom_s5, SYMMETRIES[5];
        bad_atom_a0, ANOMALOUS_CONFIGURATIONS[0];
        bad_atom_a1, ANOMALOUS_CONFIGURATIONS[1];
        bad_atom_a2, ANOMALOUS_CONFIGURATIONS[2];
        bad_atom_a3, ANOMALOUS_CONFIGURATIONS[3];
        bad_atom_a4, ANOMALOUS_CONFIGURATIONS[4];
        bad_atom_a5, ANOMALOUS_CONFIGURATIONS[5];
        bad_atom_a6, ANOMALOUS_CONFIGURATIONS[6];
        bad_atom_a7, ANOMALOUS_CONFIGURATIONS[7];
    }

    #[test]
    fn kyube() {
        assert_eq!(
            hull(vertex_set![1, 1, 1, 1, 1, 1, 1, 1])
                .interior()
                .facets()
                .count(),
            0
        );
        assert_eq!(
            hull(vertex_set![1, 1, 1, 1, 1, 1, 1, 1])
                .exterior()
                .facets()
                .count(),
            6
        );
    }

    #[test]
    fn tetrahedron() {
        let hull = hull(vertex_set![0, 1, 0, 0, 1, 1, 0, 1]);
        assert_eq!(hull.interior().facets().count(), 1);
        assert_eq!(hull.exterior().facets().count(), 3,);
    }

    #[test]
    fn fuckin_weird_dude() {
        let atom = Atom::new(vertex_set![0, 1, 1, 0, 1, 0, 0, 1]);
        let hull = atom.compound_hull();

        println!("atom: {:#}", atom.to_set());

        for axis in Axis::generator() {
            println!(
                "axis: {:?}, set: {:#}",
                axis,
                atom.to_set() & axis.to_face_set()
            );
        }

        for face in &hull.exterior().faces {
            println!("axis/face: {:?}/{:?}", face.axis(), face);
            for facet in face.facets() {
                println!("\tfacet: {:?}", facet);
            }
        }

        println!("interior facets:");
        for facet in hull.interior().facets() {
            println!("\tfacet: {:?}", facet);
        }

        assert_eq!(hull.interior().facets().count(), 4);
        assert_eq!(hull.exterior().facets().count(), 0);
    }

    #[test]
    fn join_m() {
        fn v(n: u8) -> Exact {
            Vertex::from_u8(n).to_exact()
        }

        let nz = Atom::new(vertex_set![1, 1, 0, 0, 1, 1, 1, 1]);
        let pz = Atom::new(vertex_set![1, 1, 1, 1, 1, 1, 0, 0]);

        let mut nz_hull = nz.compound_hull();
        let mut pz_hull = pz.compound_hull();
        nz_hull.join_exteriors(Axis::PosZ, &mut pz_hull);

        assert_eq!(nz_hull.interior().facets().count(), 1);
        assert_eq!(pz_hull.interior().facets().count(), 1);
        assert_eq!(nz_hull.exterior().facets().count(), 4);
        assert_eq!(pz_hull.exterior().facets().count(), 4);

        let nz_ext_facet_set = nz_hull.exterior().facets().collect::<HashSet<HullFacet>>();
        let pz_center = Exact(Point3::new(1, 1, 2));
        assert_eq!(
            nz_ext_facet_set,
            hashset! {
                HullFacet::Triangle([v(4), v(0), v(6)]),
                HullFacet::Triangle([pz_center, v(5), v(7)]),
                HullFacet::Rectangle([v(1), v(0), v(4), v(5)]),
                HullFacet::Rectangle([v(5), v(4), v(6), v(7)]),
            }
        );

        let pz_ext_facet_set = pz_hull.exterior().facets().collect::<HashSet<HullFacet>>();
        let nz_center = Exact(Point3::new(1, 1, 0));
        assert_eq!(
            pz_ext_facet_set,
            hashset! {
                HullFacet::Triangle([v(3), v(1), v(5)]),
                HullFacet::Triangle([nz_center, v(0), v(2)]),
                HullFacet::Rectangle([v(1), v(3), v(2), v(0)]),
                HullFacet::Rectangle([v(1), v(0), v(4), v(5)]),
            }
        );
    }

    #[test]
    fn n_valid_atoms() {
        assert_eq!(Atom::generator().count(), 127);
    }

    #[test]
    fn interior_winding() {
        for (_i, atom) in Atom::generator().enumerate() {
            let centroid = atom.to_set().centroid();
            // println!("atom [{:03}] w/ centroid {} {:#}", _i, centroid, atom);
            for (_j, facet) in atom.compound_hull().interior().facets().enumerate() {
                // println!("facet [{:02}]: {:?}", _j, facet);
                assert!(facet.is_normal_outwards_with_respect_to_point(&centroid));
            }
        }
    }

    #[test]
    fn exterior_winding() {
        for (_i, atom) in Atom::generator().enumerate() {
            let centroid = atom.to_set().centroid();
            // println!("atom [{:03}] w/ centroid {} {:#}", _i, centroid, atom);
            for (_j, facet) in atom.compound_hull().exterior().facets().enumerate() {
                // println!("facet [{:02}]: {:?}", _j, facet);
                assert!(facet.is_normal_outwards_with_respect_to_point(&centroid));
            }
        }
    }

    #[test]
    fn edge_filter_ramps() {
        let ramp_negz = Atom::new(vertex_set![1, 0, 1, 0, 1, 1, 1, 1]);
        let ramp_negz_coords = Vector3::new(0, 0, 0);
        let ramp_posz = Atom::new(vertex_set![1, 1, 1, 1, 1, 0, 1, 0]);
        let ramp_posz_coords = Vector3::new(0, 1, 0);

        let mut ramp_negz_hull = ramp_negz.compound_hull();
        let mut ramp_posz_hull = ramp_posz.compound_hull();
        ramp_negz_hull.join_exteriors(Axis::PosY, &mut ramp_posz_hull);

        let mut edge_filter = EdgeFilter::new();
        edge_filter.extend(
            ramp_negz_hull
                .facets()
                .map(|facet| (ramp_negz_coords, facet.translated_by(ramp_negz_coords))),
        );
        edge_filter.extend(
            ramp_posz_hull
                .facets()
                .map(|facet| (ramp_posz_coords, facet.translated_by(ramp_posz_coords))),
        );

        assert!(!edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(0, 2, 0)),
            Exact(Point3::new(2, 2, 0))
        )));
        assert!(edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(1, 2, 1)),
            Exact(Point3::new(0, 2, 0))
        )));

        assert_eq!(edge_filter.iter_extant().count(), 20);
        assert_eq!(edge_filter.iter_removed().count(), 7);
    }

    #[test]
    fn edge_filter_face_adjacent_cubes() {
        let cube = Atom::new(vertex_set![1, 1, 1, 1, 1, 1, 1, 1]);
        let cube_negy_coords = Vector3::new(0, 0, 0);
        let cube_posy_coords = Vector3::new(0, 1, 0);

        let mut cube_negy_hull = cube.compound_hull();
        let mut cube_posy_hull = cube.compound_hull();
        cube_negy_hull.join_exteriors(Axis::PosY, &mut cube_posy_hull);

        let mut edge_filter = EdgeFilter::new();
        edge_filter.extend(
            cube_negy_hull
                .facets()
                .map(|facet| (cube_negy_coords, facet.translated_by(cube_negy_coords))),
        );
        edge_filter.extend(
            cube_posy_hull
                .facets()
                .map(|facet| (cube_posy_coords, facet.translated_by(cube_posy_coords))),
        );

        assert!(!edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(0, 2, 0)),
            Exact(Point3::new(0, 2, 2))
        )));
        assert!(edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(0, 0, 0)),
            Exact(Point3::new(0, 2, 0))
        )));
        let mut edges = edge_filter.iter_extant().collect::<Vec<_>>();
        edges.sort_by_key(|e| {
            [
                e.0 .0.coords.x,
                e.0 .0.coords.y,
                e.0 .0.coords.z,
                e.1 .0.coords.x,
                e.1 .0.coords.y,
                e.1 .0.coords.z,
            ]
        });
        assert_eq!(edge_filter.iter_extant().count(), 16);
        assert_eq!(edge_filter.iter_removed().count(), 14);
    }

    #[test]
    fn edge_filter_edge_adjacent_cubes() {
        let cube = Atom::new(vertex_set![1, 1, 1, 1, 1, 1, 1, 1]);
        let cube0_coords = Vector3::new(0, 0, 0);
        let cube1_coords = Vector3::new(0, 1, 1);

        let cube0_hull = cube.compound_hull();
        let cube1_hull = cube.compound_hull();

        let mut edge_filter = EdgeFilter::new();
        edge_filter.extend(
            cube0_hull
                .facets()
                .map(|facet| (cube0_coords, facet.translated_by(cube0_coords))),
        );
        edge_filter.extend(
            cube1_hull
                .facets()
                .map(|facet| (cube1_coords, facet.translated_by(cube1_coords))),
        );

        assert!(!edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(2, 2, 2)),
            Exact(Point3::new(0, 2, 2))
        )));
        assert!(edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(0, 0, 0)),
            Exact(Point3::new(0, 2, 0))
        )));
        let mut edges = edge_filter.iter_extant().collect::<Vec<_>>();
        edges.sort_by_key(|e| {
            [
                e.0 .0.coords.x,
                e.0 .0.coords.y,
                e.0 .0.coords.z,
                e.1 .0.coords.x,
                e.1 .0.coords.y,
                e.1 .0.coords.z,
            ]
        });
        assert_eq!(edge_filter.iter_extant().count(), 22);
        assert_eq!(edge_filter.iter_removed().count(), 13);
    }

    #[test]
    fn edge_filter_face_adjacent_cubes_with_ramp() {
        let cube = Atom::new(vertex_set![1, 1, 1, 1, 1, 1, 1, 1]);
        let ramp_negz = Atom::new(vertex_set![1, 1, 1, 1, 1, 0, 1, 0]);
        let cube_negx_coords = Vector3::new(0, 0, 0);
        let cube_posx_coords = Vector3::new(1, 0, 0);
        let ramp_posz_coords = Vector3::new(0, 0, 1);

        let mut cube_negx_hull = cube.compound_hull();
        let mut cube_posx_hull = cube.compound_hull();
        let mut ramp_negz_hull = ramp_negz.compound_hull();
        cube_negx_hull.join_exteriors(Axis::PosX, &mut cube_posx_hull);
        cube_posx_hull.join_exteriors(Axis::PosZ, &mut ramp_negz_hull);

        let mut edge_filter = EdgeFilter::new();
        edge_filter.extend(
            cube_negx_hull
                .facets()
                .map(|facet| (cube_negx_coords, facet.translated_by(cube_negx_coords))),
        );
        edge_filter.extend(
            cube_posx_hull
                .facets()
                .map(|facet| (cube_posx_coords, facet.translated_by(cube_posx_coords))),
        );
        edge_filter.extend(
            ramp_negz_hull
                .facets()
                .map(|facet| (ramp_posz_coords, facet.translated_by(ramp_posz_coords))),
        );

        assert!(!edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(2, 0, 0)),
            Exact(Point3::new(2, 0, 2))
        )));
        assert!(!edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(2, 0, 2)),
            Exact(Point3::new(2, 2, 2))
        )));
        assert!(edge_filter.edge_exists(SortedPair::new(
            Exact(Point3::new(0, 0, 0)),
            Exact(Point3::new(0, 2, 0))
        )));
        let mut edges = edge_filter.iter_extant().collect::<Vec<_>>();
        edges.sort_by_key(|e| {
            [
                e.0 .0.coords.x,
                e.0 .0.coords.y,
                e.0 .0.coords.z,
                e.1 .0.coords.x,
                e.1 .0.coords.y,
                e.1 .0.coords.z,
            ]
        });
        assert_eq!(edge_filter.iter_extant().count(), 18);
        assert_eq!(edge_filter.iter_removed().count(), 18);
    }

    #[test]
    fn sorted_pair() {
        assert_eq!(
            SortedPair::new(Exact(Point3::new(0, 0, 0)), Exact(Point3::new(0, 2, 0))),
            SortedPair::new(Exact(Point3::new(0, 2, 0)), Exact(Point3::new(0, 0, 0)))
        );
    }
}
