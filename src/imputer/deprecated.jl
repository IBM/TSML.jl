################################################################################
## Deprecations for calling impute on an Imputor with a custom AbstractContext #
################################################################################
#Base.@deprecate(
#    impute(imp::Imputor, context::AbstractContext, data; kwargs...),
#    impute(data, typeof(imp)(; context=context, kwargs...))
#)
#
#Base.@deprecate(
#    impute!(imp::Imputor, context::AbstractContext, data; kwargs...),
#    impute!(data, typeof(imp)(; context=context, kwargs...))
#)
#
#Base.@deprecate impute(imp::Imputor, data) impute(data, imp)
#Base.@deprecate impute!(imp::Imputor, data) impute!(data, imp)

#####################################################################
# Deprecate all impute calls where the first argument is an Imputor #
#####################################################################
"""
    impute!(data, method::Symbol=:interp, args...; limit::Float64=0.1)

Looks up the `Imputor` type for the `method`, creates it and calls
`impute!(data, imputor::Imputor)` with it.

# Arguments
* `data`: the datset containing missing elements we should impute.
* `method::Symbol`: the imputation method to use
    (options: [`:drop`, `:fill`, `:interp`, `:locf`, `:nocb`])
* `args::Any...`: any arguments you should pass to the `Imputor` constructor.
* `limit::Float64`: missing data ratio limit/threshold (default: 0.1)
"""
function impute!(data, method::Symbol, args...; limit::Float64=0.1)
    Base.depwarn(
        """
        impute!(data, method) is deprecated.
        Please use Impute.method!(data) or impute!(data, imputor::Imputor).
        """,
        :impute!
    )
    imputor_type = imputation_methods[method]
    imputor = if length(args) > 0
        imputor_type(args...; context=Context(; limit=limit))
    else
        imputor_type(; context=Context(; limit=limit))
    end

    return impute!(data, imputor)
end

"""
    impute!(data, missing::Function, method::Symbol=:interp, args...; limit::Float64=0.1)

Creates the appropriate `Imputor` type and `Context` (using `missing` function) in order to call
`impute!(data, imputor::Imputor)` with them.

# Arguments
* `data`: the datset containing missing elements we should impute.
* `missing::Function`: the missing data function to use
* `method::Symbol`: the imputation method to use
    (options: [`:drop`, `:fill`, `:interp`, `:locf`, `:nocb`])
* `args::Any...`: any arguments you should pass to the `Imputor` constructor.
* `limit::Float64`: missing data ratio limit/threshold (default: 0.1)
"""
function impute!(data, missing::Function, method::Symbol, args...; limit::Float64=0.1)
    Base.depwarn(
        """
        impute!(data, missing, method) is deprecated. Please use impute!(data, imputor::Imputor).
        """,
        :impute!
    )
    imputor_type = imputation_methods[method]
    imputor = if length(args) > 0
        imputor_type(args...; context=Context(; is_missing=missing, limit=limit))
    else
        imputor_type(; context=Context(; is_missing=missing, limit=limit))
    end

    return impute!(data, imputor)
end

"""
    impute(data, args...; kwargs...)

Copies the `data` before calling `impute!(new_data, args...; kwargs...)`
"""
function impute(data, args...; kwargs...)
    Base.depwarn(
        """
        impute(data, args...; kwargs...) is deprecated.
        Please use Impute.method(data) or impute(data, imputor::Imputor).
        """,
        :impute
    )
    # Call `deepcopy` because we can trust that it's available for all types.
    return impute!(deepcopy(data), args...; kwargs...)
end

#################################
# Deprecate the chain functions #
#################################
"""
    chain!(data, missing::Function, imputors::Imputor...; kwargs...)

Creates a `Chain` with `imputors` and calls `impute!(imputor, missing, data; kwargs...)`
"""
function chain!(data, missing::Function, imputors::Imputor...; kwargs...)
    Base.depwarn(
        """
        chain!(data, missing, imputors...) is deprecated.
        Please use data = imp1(data) |> imp2 |> imp3
        """,
        :chain!
    )
    return chain!(data, imputors...; is_missing=missing, kwargs...)
end

"""
    chain!(data, imputors::Imputor...; kwargs...)

Creates a `Chain` with `imputors` and calls `impute!(data, imputor)`
"""
function chain!(data, imputors::Imputor...; kwargs...)
    Base.depwarn(
        """
        chain!(data, imputors...) is deprecated.
        Please use data = imp1(data) |> imp2 |> imp3
        """,
        :chain!
    )
    ctx = Context(; kwargs...)

    for imputor in imputors
        imp = typeof(imputor)(
            (isa(x, AbstractContext) ? ctx : x for x in fieldvalues(imputor))...
        )
        data = impute!(data, imp)
    end

    return data
end

"""
    chain(data, args...; kwargs...)

Copies the `data` before calling `chain!(data, args...; kwargs...)`
"""
function chain(data, args...; kwargs...)
    Base.depwarn(
        """
        chain(data, args...) is deprecated.
        Please use result = imp1(data) |> imp2 |> imp3
        """,
        :chain
    )
    # Call `deepcopy` because we can trust that it's available for all types.
    return chain!(deepcopy(data), args...; kwargs...)
end

#####################
# Misc Deprecations #
#####################
#Base.@deprecate Fill(val; kwargs...) Fill(; value=val, kwargs...)
#Base.@deprecate_binding Drop DropObs false

# This function is just used to support legacy behaviour and should be removed in a
# future release when we dropping accepting the limit kwarg to impute functions.
function _extract_context_kwargs(kwargs...)
    d = Dict{Symbol, Any}(kwargs...)
    limit = 1.0

    if haskey(d, :limit)
        limit = d[:limit]
        @warn(
            "Passing `limit` directly to impute functions is deprecated. " *
            "Please pass `context=Context(; limit=$limit)` in the future."
        )
        delete!(d, :limit)
    end

    if !haskey(d, :context)
        d[:context] = Context(; limit=limit)
    end

    return d
end
