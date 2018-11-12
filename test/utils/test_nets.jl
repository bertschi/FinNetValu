using SparseArrays

A1 = [0 0; 0 0]
A2 = [1 0; 0 1]
A3 = [1 2 3; 5 4 2; 7 2 5]
A4 = [.1 .2 .3; .3 .4 .2; .1 .2 .5]
A5 = [.5 .2 .3; .3 .4 .3; .2 .2 .3]
A6 = [.5 .9 .3; .3 .4 .3; .2 .2 .3]
spA1 = spzeros(4, 2)
spA1[1,1] = 1; spA1[1,2] = 2; spA1[2,1] = 2; spA1[2,2] = 1
spA2 = spzeros(2, 4)
spA2[1,1] = 3; spA2[1,2] = 2; spA2[2,1] = 2; spA2[2,2] = 1
spA3 = spzeros(2, 4)
spA3[1,1] = .3; spA3[1,2] = .2; spA3[2,1] = .2; spA3[2,2] = .1
spA4 = spzeros(4, 2)
spA4[1,1] = .3; spA4[1,2] = .2; spA4[2,1] = .2; spA4[2,2] = .1; spA4[3,1] = 0.5; spA4[3,2] = .7

@testset "rowsums" begin
    @test FinNetValu.rowsums(A1) == [0, 0]
    @test FinNetValu.rowsums(A2) == [1, 1]
    @test FinNetValu.rowsums(A3) == [6, 11, 14]
    @test FinNetValu.rowsums(spA1) == [3, 3, 0, 0]
    @test FinNetValu.rowsums(spA2) == [5, 3]
end

@testset "isleft_substochastic" begin
    @test FinNetValu.isleft_substochastic(A1) == true
    @test FinNetValu.isleft_substochastic(A2) == true
    @test FinNetValu.isleft_substochastic(A3) == false
    @test FinNetValu.isleft_substochastic(A4) == true
    @test FinNetValu.isleft_substochastic(A5) == true
    @test FinNetValu.isleft_substochastic(A6) == false
    @test FinNetValu.isleft_substochastic(spA1) == false
    @test FinNetValu.isleft_substochastic(spA2) == false
    @test FinNetValu.isleft_substochastic(spA3) == true
    @test FinNetValu.isleft_substochastic(spA4) == true
end