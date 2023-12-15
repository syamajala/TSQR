import "regent"

local c = regentlib.c

local helper_exp = {}

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~        input arguments                 ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--function to check if a provided input file is valid
terra helper_exp.file_exists(filename : rawstring)
  var file = c.fopen(filename, "r")
  if file == nil then return false end
  c.fclose(file)
  return true
end

--task to initialize the matrix region if no input file is provided
task helper_exp.initialize(x : int, m : int, matrix : region(ispace(int2d), double))
where reads(matrix), writes(matrix)
do

  fill(matrix, 1.0)

  for i in matrix do
    var x_shift = i.x - x*m
    if i.y == x_shift then
      matrix[i] += i.y
    end
  end

end

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~        domain coloring                 ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--task to partition the matrix for matrix multiplication
task helper_exp.color_matrix(blocks     : int,
                             matrix     : region(ispace(int2d), double),
                             mat_colors : ispace(int1d))
  var t : transform(2, 1)

  t[{0, 0}] = blocks
  t[{1, 0}] = 0

  var e = rect2d{ int2d{0, 0}, int2d{blocks-1, blocks-1} }
  return restrict(disjoint, complete, matrix, t, e, mat_colors)
end

--task to partition a 2D region by processor id
task helper_exp.processor_matrix_part(dim_x      : int,
                                      levels     : int,
                                      matrix     : region(ispace(int2d), double),
                                      mat_colors : ispace(int1d))

  var start_point = matrix.bounds.lo.x
  var length : int = dim_x

  var matrix_coloring = c.legion_multi_domain_point_coloring_create()

  --color the matrix
  for j = 0,levels do
    for i in mat_colors do
      c.legion_multi_domain_point_coloring_color_domain(matrix_coloring, i,
                                                        rect2d {lo = {x = start_point, y = matrix.bounds.lo.y},
                                                                hi = {x = (start_point + length - 1), y = matrix.bounds.hi.y}})

      start_point += length

    end
  end

  var mat_part = partition(disjoint, matrix, matrix_coloring, mat_colors)
  c.legion_multi_domain_point_coloring_destroy(matrix_coloring)

  return mat_part

end

--task to partition a 2D region by processor id
-- __demand(__leaf)
-- task processor_matrix_part(dim_x  : int,
--                            bounds : rect2d,
--                            lr     : region(ispace(int2d), rect2d))
-- where writes(lr) do
--   var start_point = bounds.lo.x
--   var length : int = dim_x

--   --color the matrix
--   for j = 0, lr.ispace.bounds.hi.x+1 do
--     for i = 0, lr.ispace.bounds.hi.y+1 do
--       lr[int2d{j, i}] = rect2d {lo = {x = start_point, y = bounds.lo.y},
--                                 hi = {x = (start_point + length - 1), y = bounds.hi.y}}
--       start_point += length
--     end
--   end
-- end

--task to create a 1D partition of a 2D region
task helper_exp.equal_matrix_part(dim_x      : int,
                                  matrix     : region(ispace(int2d), double),
                                  mat_colors : ispace(int1d))
  var t : transform(2, 1)

  t[{0, 0}] = dim_x
  t[{1, 0}] = 0

  var e = rect2d{ int2d{0, 0}, int2d{dim_x-1, dim_x-1} }

  return restrict(disjoint, complete, matrix, t, e, mat_colors)
end

--task to create a 1D partition of a 2D region
task helper_exp.copy_matrix_part(dim_x : int, offset : int, matrix : region(ispace(int2d), double), mat_colors : ispace(int1d))
  var t : transform(2, 1)

  t[{0, 0}] = dim_x
  t[{1, 0}] = 0

  var e = rect2d{ int2d{offset, 0}, int2d{offset + (dim_x/2) - 1, dim_x - 1} }

  return restrict(disjoint, complete, matrix, t, e, mat_colors)
end

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~~~         matrix extractions             ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

--task to copy the upper diagonal results of the dgeqrf BLAS subroutine to corresponding R region
-- task helper_exp.get_R_matrix(p : int, m_vec : region(ispace(int1d), int), n : int, matrix : region(ispace(int2d), double), R_matrix : region(ispace(int2d), double))

-- where reads(matrix, m_vec), writes(R_matrix)
-- do
--   var r_point : int2d
--   var x_shift : int
--   var offset : int = 0

--   for j = 0, p do
--     offset += m_vec[j]
--   end

--   for i in matrix do
--     x_shift = i.x - offset
--     if i.y >= x_shift then
--       r_point = {x = i.x - offset + p*n, y = i.y}
--       R_matrix[r_point] = matrix[i]
--     end
--   end
-- end

--task to copy the computed Q matrix from the dorgqr BLAS subroutine to the processor's "unique" region
task helper_exp.get_Q_matrix(Q_matrix : region(ispace(int2d), double), temp_matrix : region(ispace(int2d), double))

where reads(temp_matrix), writes(Q_matrix)
do
  var x_shift : int = Q_matrix.bounds.lo.x - temp_matrix.bounds.lo.x

  for i in temp_matrix do
    Q_matrix[{x = i.x + x_shift, y = i.y}] = temp_matrix[i]
  end
end

--task to copy a matrix from one region to another
task helper_exp.copy_function(source_region : region(ispace(int2d), double), destination_region : region(ispace(int2d), double))
where reads(source_region), writes(destination_region)
do

  var x_shift : int = destination_region.bounds.lo.x - source_region.bounds.lo.x

  for i in source_region do
    destination_region[{x = i.x + x_shift, y = i.y}] = source_region[i]
  end

end

return helper_exp
