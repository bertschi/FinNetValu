"""
    constantly(val)

Create constant function that returns `val` when called.
"""
constantly(val) = (x...) -> val

"""
    calm(f, n)

Create a new function that behaves like `f()` but returns the same
value for `n` successive calls, i.e. `f` is invoked once for every `n`
calls of `calm(f, n)`.  
"""
function calm(f, n)
    i = 0
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
