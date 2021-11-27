use std::fmt;

use arrayvec::ArrayVec;
use bitvec::prelude::*;
use nalgebra::*;

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
#[derive(Debug, Clone, Copy)]
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

        let axis_bits = axis.to_bits();
        bits[4..7].copy_from_bitslice(&axis_bits[0..3]);

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
            if !self.axis().requires_winding_flip() {
                verts.reverse();
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
                    [face_verts[v1], face_verts[v2], face_verts[v3]].map(Vertex::to_exact),
                )
            } else {
                HullFacet::Triangle(
                    [face_verts[v2], face_verts[v1], face_verts[v3]].map(Vertex::to_exact),
                )
            }
        };

        let smol = |facet: usize| {
            let v1 = facet;
            let v2 = (v1 + 1) & 0b11;

            // flip winding order
            if self.axis().requires_winding_flip() {
                HullFacet::Triangle([center, face_verts[v1].to_exact(), face_verts[v2].to_exact()])
            } else {
                HullFacet::Triangle([center, face_verts[v2].to_exact(), face_verts[v1].to_exact()])
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
#[derive(Debug, Clone)]
pub struct InteriorHull {
    facets: ArrayVec<HullFacet, 8>,
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct CompoundHull {
    interior: InteriorHull,
    exterior: ExteriorHull,
}

impl CompoundHull {
    /// Calculate the initial compound hull (before join operations) of an atom.
    #[inline]
    pub fn new(atom: Atom) -> Self {
        Self {
            interior: InteriorHull::new(atom),
            exterior: ExteriorHull::new(atom),
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
    /// Find the facet's normal.
    #[inline]
    pub fn normal(self) -> UnitVector3<f32> {
        let (Self::Triangle([a, b, c]) | Self::Rectangle([a, b, c, _])) = self;
        let (p0, p1, p2) = (a.to_f32(), b.to_f32(), c.to_f32());
        let v1 = p0 - p1;
        let v2 = p2 - p1;
        UnitVector3::new_normalize(v1.cross(&v2))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{vertex_set, FACES};
    use maplit::hashset;
    use nalgebra::*;
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
                HullFacet::Triangle([v(0), v(4), v(6)]),
                HullFacet::Triangle([pz_center, v(7), v(5)]),
                HullFacet::Rectangle([v(5), v(4), v(0), v(1)]),
                HullFacet::Rectangle([v(7), v(6), v(4), v(5)]),
            }
        );

        let pz_ext_facet_set = pz_hull.exterior().facets().collect::<HashSet<HullFacet>>();
        let nz_center = Exact(Point3::new(1, 1, 0));
        assert_eq!(
            pz_ext_facet_set,
            hashset! {
                HullFacet::Triangle([v(1), v(3), v(5)]),
                HullFacet::Triangle([nz_center, v(2), v(0)]),
                HullFacet::Rectangle([v(1), v(0), v(2), v(3)]),
                HullFacet::Rectangle([v(5), v(4), v(0), v(1)]),
            }
        );
    }

    #[test]
    fn n_valid_atoms() {
        assert_eq!(Atom::generator().count(), 127);
    }

    #[test]
    fn interior_winding() {
        for (i, atom) in Atom::generator().enumerate() {
            let centroid = atom.to_set().centroid();
            println!("atom [{:03}] w/ centroid {} {:#}", i, centroid, atom);
            for (j, facet) in atom.compound_hull().interior().facets().enumerate() {
                println!("facet [{:02}]: {:?}", j, facet);
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
}
