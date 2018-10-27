using FinNetValu
using LinearAlgebra

suzuki = XOSModel(4,
                  [[0.0 0.2 0.3 0.1];
                   [0.2 0.0 0.2 0.1];
                   [0.1 0.1 0.0 0.3];
                   [0.1 0.1 0.1 0.0]],
                  [[0.0 0.0 0.1 0.0];
                   [0.0 0.0 0.0 0.1];
                   [0.1 0.0 0.0 0.1];
                   [0.0 0.1 0.0 0.0]],
                  LinearAlgebra.I,
                  [0.8, 0.8, 0.8, 0.8])

Aᵉ = [2.0, 0.5, 0.6, 0.6]

println(suzuki)
println(value(suzuki, Aᵉ))
