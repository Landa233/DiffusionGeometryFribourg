git mv diffusion_geometry/src/diffusion_process.py src/core/diffusion/
git mv diffusion_geometry/classes/markov_triples.py src/core/diffusion/
git mv diffusion_geometry/src/regularise.py src/core/diffusion/
git mv diffusion_geometry/classes/symmetric_kernel.py src/core/diffusion/
# Mapping main.py to diffusion_geometry.py as per typical refactor logic
git mv diffusion_geometry/classes/main.py src/core/geometry/diffusion_geometry.py 

# 3. Move files to Operators
git mv diffusion_geometry/src/derivative.py src/operators/differential_operators/
git mv diffusion_geometry/src/hessian.py src/operators/differential_operators/
git mv diffusion_geometry/src/laplacian.py src/operators/differential_operators/
git mv diffusion_geometry/src/levi_civita.py src/operators/differential_operators/
git mv diffusion_geometry/src/lie_bracket.py src/operators/differential_operators/

git mv diffusion_geometry/classes/operators/bilinear.py src/operators/types/
git mv diffusion_geometry/classes/operators/direct_sum.py src/operators/types/
git mv diffusion_geometry/classes/operators/linear.py src/operators/types/

# 4. Move files to Tensors
git mv diffusion_geometry/classes/tensor_spaces/base.py src/tensors/base_tensor/base_tensor_space.py
git mv diffusion_geometry/classes/tensors/base.py src/tensors/base_tensor/base_tensor.py
git mv diffusion_geometry/src/metric_gram.py src/tensors/base_tensor/metric_gram.py

git mv diffusion_geometry/classes/tensors/direct_sum_element.py src/tensors/direct_sum/
git mv diffusion_geometry/classes/tensor_spaces/direct_sum_space.py src/tensors/direct_sum/

git mv diffusion_geometry/classes/tensor_spaces/form_space.py src/tensors/forms/
git mv diffusion_geometry/classes/tensors/form.py src/tensors/forms/

git mv diffusion_geometry/classes/tensor_spaces/function_space.py src/tensors/functions/
git mv diffusion_geometry/classes/tensors/function.py src/tensors/functions/

git mv diffusion_geometry/classes/tensor_spaces/tensor02_space.py src/tensors/tensor02/
git mv diffusion_geometry/classes/tensors/tensor02.py src/tensors/tensor02/

git mv diffusion_geometry/classes/tensor_spaces/tensor02sym_space.py src/tensors/tensor02sym/
git mv diffusion_geometry/classes/tensors/tensor02sym.py src/tensors/tensor02sym/

git mv diffusion_geometry/classes/tensor_spaces/vector_field_space.py src/tensors/vector_fields/
git mv diffusion_geometry/classes/tensors/vector_field.py src/tensors/vector_fields/

# 5. Move files to Utils
git mv diffusion_geometry/classes/tensors/basis_conversions.py src/utils/
git mv diffusion_geometry/src/basis_utils.py src/utils/
git mv diffusion_geometry/classes/tensors/batch_utils.py src/utils/

git mv diffusion_geometry/classes/cache.py src/core/geometry/cache.py  

git mv diffusion_geometry/src/carre_du_champ.py src/core/diffusion/carre_du_champ.py

# 6. Optional: Cleanup empty old directories
# find diffusion_geometry -type d -empty -delete

echo "Reorganization complete!"