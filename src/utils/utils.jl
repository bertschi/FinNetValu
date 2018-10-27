"""
    constantly(val)

Create constant function that returns `val` when called.
"""
constantly(val) = (x...) -> val

"""
    fillrows(v)

Fills a square matrix by repeating a vector `v` rowwise.
"""
function fillrows(v::AbstractVector)
    D = length(v)
    repeat(reshape(v, 1, D), D, 1)
end

"""
    calm(f, n)

Create a new function that behaves like `f()` but returns the same
value for `n` successive calls, i.e. `f` is invoked once for every `n`
calls of `calm(f, n)`.  
"""
function calm(f, n)
    i = 1
    val = f()
    function calmed()
        if i == n
            i = 1
            val = f()
        else
            i += 1
        end
        val
    end
    calmed
end
