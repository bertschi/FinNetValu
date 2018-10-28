# Financial network valuation models

```@meta
CurrentModule = FinNetValu
```

Many financial network models used in the study of systemic risk can
be considered as network valuations. This library aims to support a
range of such models. 

## Basic interface

A network valuation model is expected to support the following
interface:

```@docs
valuation!
```

```@docs
valuation
```

```@docs
init
```

```@docs
valuefunc
```

## API

Based on these the self-consistent network value is computed as the
fixed point `x = valuation(net, x, a)`. By default NLsolve is used.

```@docs
fixvalue
```

Furthermore, using AutoDiff the Jacobian matrix at the fixed point is
computed via the inverse function theorem.

```@docs
fixjacobian
```

## Implemented models

```@docs
XOSModel
```

```@docs
NEVAModel
```

```@docs
EisenbergNoeModel
```
