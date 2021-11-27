# Soft-Edge: efficient bithackery for making 3D collision meshes out of grids and stacked tile maps

*indecipherable wailing guitar noises and tape echo sounds* - Wata (Boris - Soft Edge)

`soft-edge` is a crate which provides utilities for dealing with 3D grids where the cells are
colliders defined as convex hulls of subsets of the unit cube.

Things it gives you:
- `Vertex`, representing a vertex of the unit cube as an index in `0..8`
- `VertexSet`, representing a subset of the vertices of the unit cube as a 1-byte bitset
- `Atom`, representing a valid convex non-coplanar subset of the vertices of the unit cube
- `CompoundHull`, representing a clippable convex hull of an `Atom`
- `HullFacet`, representing a polygon of a potentially clipped compound hull of an atom. Hull facets
  calculated from compound hulls of atoms will always be wound CCW.

# Why?

If you've ever tried building a game physics system with a tile-based map before, you've
probably run into the "crack problem". This is where if you have two adjacent tiles, things
which should smoothly move over the "crack" between them (which mathematically does not exist
because the vertices of these two tiles are shared) instead get caught on the crack, because
during collision detection the object is moved into a tile by some force and ends up spuriously
colliding with an unexposed edge of one tile.

`soft-edge`'s `CompoundHull` type represents a sort of encoded tile collider with a
`join_exteriors` method that erases any of these unexposed edges between two tiles, leaving you
with the surface polygons that can't cause the crack problem. In addition, these clipping
operations are extremely fast, requiring only three bitwise operations (maybe some more because
they're done on only a subset of the bits of a thing, but still, it's fast.)

# Does it work?

Uh, I think so. All the tests pass, at least?

Many things in this crate are specialized/brute forced/handwritten as lookup tables. I've fixed
all the bugs I could find, but I'm sure there are more hiding somewhere, probably in mistakes in
bit-significance or something. Though most bitwise ops are done through the `bitvec` crate
exactly to avoid this as much as possible.

# Vertex numbering

Vertices are numbered as follows:

```
  v3      v7
    *----*
v2 /| v6/|
  *----* |   +Y
  | *--|-*   ^ ^ +Z
  |/v1 |/v5  |/
  *----*     +--> +X
 v0    v4
```

For more convenience, a conversion between vertex index and unit cube coordinates:

```
// v0 is the origin.
v0 <=> {0, 0, 0} <=> 0b000
v1 <=> {0, 0, 1} <=> 0b001
v2 <=> {0, 1, 0} <=> 0b010
v3 <=> {0, 1, 1} <=> 0b011
v4 <=> {1, 0, 0} <=> 0b100
v5 <=> {1, 0, 1} <=> 0b101
v6 <=> {1, 1, 0} <=> 0b110
v7 <=> {1, 1, 1} <=> 0b111
```

This is derived as a bitwise representation `0bXYZ` where the X bit represents whether or not
the X axis is `1`, Y bit is whether or not the Y axis is `1`, etc.

# "Exact" coordinates for vertex index deduplication

In order to facilitate vertex deduplication (for an indexed triangle mesh, for example) most of
the output coordinates for things like the `CompoundHull`'s facets (`HullFacet`) are
presented using the `Exact` type. This wraps `nalgebra`'s `Point3<i32>`, and is essentially
just the same coordinates you would expect but scaled by a factor of 2 on every axis. This
allows us to represent centroids of faces with no possibility of error, and also provides a
hashable type which can be used to build a set mapping vertices to indices.

# Generally speaking, how do you expect me to use this?

1. Encode your collision map as a grid of bytes, which correspond to valid `Atom`s.
   Deserialize these bytes into `VertexSet`s using `VertexSet::from_u8`, and try to
   construct atoms from them with `Atom::try_new`.
2. Construct a three-dimensional grid consisting of the `CompoundHull`s of the atom grid elements,
   and then for every hull in the grid, `join` it with its neighbors on the proper axes.
3. Extract facets of these joined compound hulls as `HullFacet`s, and then construct your
   collision geometry with them. This will end up as a very much non-convex mesh, which you may
   want to decompose into convex sub-meshes (but don't have to, if you do collisions directly on
   polygons and only allow contact normals which penetrate through their "surface" side.)

`soft-edge` attempts to produce hull facets with the correct winding order (CCW), which should allow
you to easily calculate their normals. If it does not, this is a bug, and must be fixed.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
