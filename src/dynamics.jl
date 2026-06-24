# ─────────────────────────────────────────────────────────────────────────────
#  ODE right-hand-side machinery
# ─────────────────────────────────────────────────────────────────────────────

"Return the correct RHS function for a given application"
function make_rhs_function(application::String)
    application == "RP-RW" ? rhs_RW! :
    application == "RP-IW" ? rhs_basic! :
    error("Unknown application type: $application")
end

"Right-hand side for the rational-surface + resistive-wall (RP-RW) system"
function rhs_RW!(dydt, y, t, Control2::Float64, C1::Float64, thEF::Float64, ode_params::ODEparams, n_tor::Int, control_type::Symbol)
    n0 = n_tor
    DeltaW = ode_params.DeltaW
    rt = ode_params.rat_surface
    l21 = ode_params.l21
    l12 = ode_params.l12
    l32 = ode_params.l32
    Tt_Tw = ode_params.Taut_Tauw
    mu = ode_params.mu
    I = ode_params.Inertia

    alpha, errF, Deltat = control_adjustments(ode_params, Control2, control_type)

    psi, theta, Om, psiW, thW = y

    dydt[1] = Deltat * psi * (1.0 + alpha * abs(psi)) + l21 * psiW * cos(theta - thW)
    dydt[2] = -n0 * Om - l21 * psiW * sin(theta - thW) / psi
    dydt[3] = (rt * l21 * psiW * psi * sin(theta - thW) + mu * (C1 - Om)) / I
    dydt[4] = Tt_Tw * (DeltaW * psiW + l12 * psi * cos(theta - thW) + l32 * errF * sin(thEF - thW))
    dydt[5] = Tt_Tw * (l12 * psi * sin(theta - thW) - l32 * errF * cos(thEF - thW)) / psiW
end

"Right-hand side for the rational-surface + ideal-wall (RP-IW) system"
function rhs_basic!(dydt, y, t, Control2::Float64, C1::Float64, thEF::Float64, ode_params::ODEparams, n_tor::Int, control_type::Symbol)
    n0 = n_tor
    rt = ode_params.rat_surface
    l21 = ode_params.l21
    mu = ode_params.mu
    I = ode_params.Inertia

    alpha, errF, Deltat = control_adjustments(ode_params, Control2, control_type)

    psi, theta, Om = y

    dydt[1] = Deltat * psi * (1.0 + alpha * abs(psi)) + l21 * errF * sin(thEF - theta)
    dydt[2] = -n0 * Om - l21 * errF * cos(thEF - theta) / psi
    dydt[3] = (rt * l21 * errF * psi * cos(thEF - theta) + mu * (C1 - Om)) / I
end

"Random initial condition inside the hyper-cube `dims`, for the given application"
function make_initial_condition(dims::Vector{Float64}, application::String)
    if application == "RP-RW"
        # 5D system
        return [
            rand() * (dims[1] - 0.001) + 0.001,   # y1
            rand() * 2π - π,                      # y2
            rand() * (dims[2] - 0.001) + 0.001,   # y3
            rand() * (dims[3] - 0.001) + 0.001,   # y4
            rand() * 2π - π                       # y5
        ]

    elseif application == "RP-IW"
        # 3D system
        return [
            rand() * (dims[1] - 0.001) + 0.001,   # y1
            rand() * 2π - π,                      # y2
            rand() * (dims[2] - 0.001) + 0.001    # y3
        ]

    else
        error("Unknown application: $application")
    end
end

"Wrap an `rhs!` function into the `(du, u, p, t)` signature expected by `ODEProblem`"
function make_ode_func(rhs!)
    return function (du, u, p, t)
        # p layout: (C2, C1, EFphase, ode_params, n_tor, control_type)
        # C2 = swept control param (EF / Δ′ / α); C1 = rotation frequency
        C2, C1, EFphase, ode_params, n_tor, control_type = p
        rhs!(du, u, t, C2, C1, EFphase, ode_params, n_tor, control_type)
    end
end


# ─────────────────────────────────────────────────────────────────────────────
#  Control-type resolution — single source of truth for control branching
# ─────────────────────────────────────────────────────────────────────────────

"""
    resolve_control(ode_params, control_type, C2) → NamedTuple

Scalar resolver: map a single raw Control2 value to concrete physical quantities.
This is the *only* place in the code where control-type branching lives.

Returns `(; Deltat, eps, alpha, DeltatRW)`.
- `Deltat`    : effective tearing stability index
- `eps`       : error-field amplitude
- `alpha`     : nonlinear saturation parameter
- `DeltatRW`  : effective resistive-wall stability  (Deltat − l21·l12/ΔW)
"""
function resolve_control(ode_params::ODEparams, control_type::Symbol, C2::Real)
    Deltat = control_type == :EF           ? ode_params.stability_index :
             control_type == :LinStab      ? Float64(C2) :
             control_type == :NLsaturation ? ode_params.stability_index :
             error("Unknown control_type: $control_type")

    eps    = control_type == :EF           ? Float64(C2) :
             control_type == :LinStab      ? ode_params.error_field :
             control_type == :NLsaturation ? ode_params.error_field :
             error("Unknown control_type: $control_type")

    alpha  = control_type == :NLsaturation ? Float64(C2) : ode_params.saturation_param

    DeltatRW = Deltat - ode_params.l21 * ode_params.l12 / ode_params.DeltaW

    # This check is only meaningful for :LinStab, where Deltat=C2 is the swept
    # quantity and DeltatRW must stay negative for the RP-RW system to remain
    # weakly stable. For :EF/:NLsaturation, Deltat=stability_index is fixed and
    # DeltatRW reduces to RPRW_stability_index — not a "sweep range" to validate.
    if control_type == :LinStab && DeltatRW > 0
        println("*** ALERT: You set up a case with an unstable RP-RW mode! ***")
        error("Deltat_RW > 0 for your range of TM stability values ***")
    end

    return (; Deltat, eps, alpha, DeltatRW)
end


"""
    resolve_control(ode_params, control_type) → NamedTuple

Vector resolver: operates over the full Control1/Control2 grid stored in `ode_params`.
Returns `(; X, Y, Deltat, eps, alpha, DeltatRW)` — all `Vector{Float64}`.
"""
function resolve_control(ode_params::ODEparams, control_type::Symbol)
    X    = ode_params.Control2   # swept control axis
    Y    = ode_params.Control1   # rotation-frequency axis
    n    = length(X)

    Deltat = control_type == :EF           ? fill(ode_params.stability_index, n) :
             control_type == :LinStab      ? copy(X) :
             control_type == :NLsaturation ? fill(ode_params.stability_index, n) :
             error("Unknown control_type: $control_type")

    eps    = control_type == :EF           ? copy(X) :
             control_type == :LinStab      ? fill(ode_params.error_field, n) :
             control_type == :NLsaturation ? fill(ode_params.error_field, n) :
             error("Unknown control_type: $control_type")

    alpha  = control_type == :NLsaturation ? copy(X) : fill(ode_params.saturation_param, n)

    DeltatRW = @. Deltat - ode_params.l21 * ode_params.l12 / ode_params.DeltaW

    return (; X, Y, Deltat, eps, alpha, DeltatRW)
end

"Apply control_type adjustments — delegates to resolve_control"
function control_adjustments(ode_params::ODEparams, C2::Float64, control_type::Symbol)
    sc = resolve_control(ode_params, control_type, C2)
    return sc.alpha, sc.eps, sc.Deltat
end


# ─────────────────────────────────────────────────────────────────────────────
#  Shared low-level ODE solving and normalization
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_ODEs(ode_params, application, n_tor, control_type, t_final, time_steps, C1, C2; full_output=false)

Solve the reduced-order ODE system for one (C1, C2) point on the control grid.

- `application`  : "RP-RW" or "RP-IW" — selects the RHS function
- `n_tor`        : toroidal mode number
- `control_type`  : :EF, :LinStab, or :NLsaturation — selects control-parameter branching
- `t_final`       : final integration time
- `time_steps`    : number of saved time steps when `full_output=true`
- `C1`            : rotation-frequency control value
- `C2`            : swept control value (EF / Δ′ / α, depending on `control_type`)
- `EFphase`       : EF phase from actor.par

When `full_output=false` (default), returns only the final state vector
(`sol.u[end]`) — used for grid scans. When `full_output=true`, returns the
full time-resolved `ODESolution` — used for single-case time traces.
"""
function solve_ODEs(ode_params::ODEparams, application::String, n_tor::Int, EFphase::Float64, control_type::Symbol,
                     t_final::Real, time_steps::Int, C1::Float64, C2::Float64; full_output::Bool=false)
    rhs!     = make_rhs_function(application)
    ode_rhs! = make_ode_func(rhs!)
    y0 = make_initial_condition(ode_params.hyper_cube_dims, application)

    # rhs! expects (C2, C1, EFphase, ...) as positional args — swap once here so
    # all callers use the natural (C1, C2) convention
    p = (C2, C1, EFphase, ode_params, n_tor, control_type)
    tspan = (0.0, Float64(t_final))

    prob = ODEProblem(ode_rhs!, y0, tspan, p)

    if full_output
        # Return the full time-resolved solution (used for time-dependent plots)
        tsave = range(0.0, Float64(t_final); length=time_steps)
        sol = solve(prob, Tsit5(); saveat=tsave, reltol=1e-8, abstol=1e-10)
        return sol
    else
        # Only the final state is needed (e.g. grid scans for classification/NN)
        sol = solve(prob, Tsit5(); saveat=Float64(t_final), reltol=1e-8, abstol=1e-10)
        return sol.u[end]
    end
end


"""
normalize_ode_results(results, ode_params, eps_vec, C1_vec, control_type)

Normalize the final solutions from ODE runs.

- `results` may be:
    * a single solution vector (Vector{Float64}), or
    * a Vector of solution vectors (Vector{Vector{Float64}}).
- `eps_vec` (Control2 values) and `C1_vec` (Control1 values) may be:
    * single Float64 values (if results is one vector), or
    * Vector{Float64} of the same length as results.

Normalization:
    psiN  = final_sol[1] * (Deltat * DeltaW - l12 * l21) / (l32 * l21 * eps)
    psiwN = final_sol[4] * (Deltat * DeltaW - l12 * l21) / (l32 * abs(Deltat) * eps)
    OmN   = final_sol[3] / C1
"""
function normalize_ode_results(results, ode_params::ODEparams, C2_vec, C1_vec, control_type)
    # Extract parameters
    l12    = ode_params.l12
    l21    = ode_params.l21
    l32    = ode_params.l32
    DeltaW = ode_params.DeltaW

    # Normalization for one solution — control branching handled by resolve_control
    function normalize_one(final_sol::AbstractVector{<:Real}, C2::Float64, C1::Float64)
        sc       = resolve_control(ode_params, control_type, C2)
        Deltat   = sc.Deltat
        eps      = sc.eps
        alpha    = sc.alpha
        DeltatRW = sc.DeltatRW

        if length(final_sol) == 5 # RP-RW layout
            psit    = final_sol[1]
            theta_t = mod(final_sol[2], 2π)
            OmN     = final_sol[3] / C1
            psiw    = final_sol[4]
            theta_w = mod(final_sol[5], 2π)
            rho = abs(DeltatRW / Deltat)

            if iszero(alpha)  # linear regime: saturation_param set to 0. when NL_saturation=false
                num     = abs(Deltat * DeltaW) - l12 * l21
                psitMax = l32 * l21 * eps / num
                psiwMax = l32 * abs(Deltat) * eps / abs(DeltatRW * DeltaW)
                psiwMin = l32 * eps / abs(DeltaW)
            else              # NL saturation active
                psitMax = -(DeltatRW + sqrt(DeltatRW^2 + 4*alpha*l21*l32*eps*Deltat/DeltaW)) /
                           (2*alpha*Deltat)
                psiwMax = -(l32*eps + l12*psitMax) / DeltaW
                psiwMin = l32 * eps / abs(DeltaW)
            end

            psiN  = abs(psit / psitMax)
            psiwN = abs(psiw / psiwMax)
            psiwN = (psiwN - rho) / (1 - rho)
            #psiwN = abs((psiw - psiwMin) / (psiwMax - psiwMin))

            return [psiN, theta_t, OmN, psiwN, theta_w]

        elseif length(final_sol) == 3 # RP-IW layout
            psit    = final_sol[1]
            theta_t = mod(final_sol[2], 2π)
            OmN     = final_sol[3] / C1

            if iszero(alpha)  # linear regime: saturation_param set to 0. when NL_saturation=false
                psitMax = l21 * eps / abs(Deltat)
            else              # NL saturation active
                psitMax = (-1.0 + sqrt(1.0 - 4*l21*alpha*eps/Deltat)) / (2*alpha)
            end

            psiN = abs(psit / psitMax)

            return [psiN, theta_t, OmN]

        else
            throw(ArgumentError("Unexpected final_sol length: $(length(final_sol))"))
        end
    end

    if isa(results, AbstractVector{<:Real}) && isa(C2_vec, Real) && isa(C1_vec, Real)
        # Single case
        return normalize_one(results, C2_vec, C1_vec)

    elseif isa(results, AbstractMatrix{<:Real}) &&
           isa(C2_vec, AbstractVector{<:Real}) &&
           isa(C1_vec, AbstractVector{<:Real})
        # Many cases — results is (N*M × n_states), iterate over rows
        size(results, 1) == length(C2_vec) == length(C1_vec) ||
            throw(ArgumentError("results, C2_vec, and C1_vec must all have the same length"))
        return reduce(vcat, (normalize_one(sol, e, c1)'
              for (sol, e, c1) in zip(eachrow(results), C2_vec, C1_vec)))

    else
        throw(ArgumentError("Input types do not match expected patterns"))
    end
end


# ─────────────────────────────────────────────────────────────────────────────
#  Grid scans, classification, and analytic bifurcation bounds
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_grid(ode_params, application, n_tor, control_type, t_final, time_steps) → Matrix{Float64}

Solve the reduced-order ODE system over the full `(Control1, Control2)` grid
stored in `ode_params`, in parallel via `pmap`. Returns a `(N*M × n_states)`
matrix of final states (one row per grid point, in `Control1`/`Control2` order).
"""
function solve_grid(ode_params::ODEparams, application::String, n_tor::Int, EFphase::Float64,
                    control_type::Symbol,t_final::Real, time_steps::Int)
    println("Solving the FULL system, this may take a few seconds")

    # Control1 = C1 (rotation, Y-axis), Control2 = C2 = EF/Δ′/α (X-axis)
    control1 = ode_params.Control1
    control2 = ode_params.Control2
    inputs = collect(zip(control1, control2))   # (C1, C2) pairs — natural order

    # Shrink what we ship to workers by clearing large control arrays
    ode_params_send = deepcopy(ode_params)
    ode_params_send.Control1 = Float64[]
    ode_params_send.Control2 = Float64[]

    # Parallel map over the grid, returning final states as a (N*M × n_states) matrix
    finals = pmap(inputs) do (C1, C2)
        solve_ODEs(ode_params_send, application, n_tor, EFphase, control_type, t_final, time_steps, C1, C2)
    end

    return Matrix(reduce(hcat, finals)')  # Vector{Vector} → Matrix{Float64} (N*M × n_states)
end


"""
    solve_and_classify(ode_params, application, n_tor, control_type, t_final, time_steps,
                        NL_saturation, grid_size) → LockingResults

Solve the full control grid (`solve_grid`), normalize the results, classify
into "locked"/"unlocked" via k-means on `(psiN, OmN)`, and (unless NL
saturation is active) compute the analytic bifurcation boundary.
"""
function solve_and_classify(ode_params::ODEparams, application::String, n_tor::Int, EFphase::Float64,
                            control_type::Symbol,t_final::Real, time_steps::Int, NL_saturation::Bool, grid_size::Int)
    control1 = ode_params.Control1
    control2 = ode_params.Control2
    ode_sols = solve_grid(ode_params, application, n_tor, EFphase, control_type, t_final, time_steps)
    norm_sols = normalize_ode_results(ode_sols, ode_params, control2, control1, control_type)

    ## classify normalized solutions
    R = hcat(norm_sols[:,1], norm_sols[:,3])
    kmc = kmeans(R', 2)
    locking_labels = kmc.assignments

    # Fix label polarity: the point with maximum OmN (col 3) is always unlocked.
    # k-means labels are {1,2}; we want that point to carry label 1 so that
    # after the {1,2}→{0,1} shift in prepare_nn_data it maps to 0 (unlocked).
    if locking_labels[argmax(norm_sols[:, 3])] != 1
        locking_labels = 3 .- locking_labels   # flip 1↔2
    end

    bifurcation_bounds = calculate_bifurcation_bounds(ode_params, application, control_type, n_tor, grid_size)

    return LockingResults(
        ode_sols,
        nothing,
        norm_sols,
        locking_labels,
        bifurcation_bounds
    )
end


"""
    calculate_bifurcation_bounds(ode_params, application, control_type, n_tor, grid_size) → Matrix{Float64}

Compute the bifurcation boundary over the full control grid.

**Linear case (α = 0):** analytic cubic discriminant; negative entries mark
the parameter region where hysteresis/locking is possible.

**NL saturation case (α ≠ 0):** the steady-state equation becomes degree 8 in
ψ, so the boundary is located numerically: for each grid point, `Roots.find_zeros`
counts the positive real roots of the steady-state residual.  The returned matrix
contains -1.0 where ≥ 2 roots exist (multiple equilibria → locking possible) and
+1.0 elsewhere.

Two wall configurations are supported via `application`:
  - "RP-RW"  (5th-order system): uses DeltatRW and the full RW geometry factors.
  - "RP-IW"  (3rd-order system): uses Deltat directly, no wall inductance factors.
"""
function calculate_bifurcation_bounds(ode_params::ODEparams, application::String, control_type::Symbol,
                                       n_tor::Int, grid_size::Int)
    n0     = n_tor
    mu     = ode_params.mu
    rt     = ode_params.rat_surface
    l21    = ode_params.l21
    l32    = ode_params.l32
    DeltaW = ode_params.DeltaW

    # resolve_control handles all control-type branching; wall branching is separate
    sc = resolve_control(ode_params, control_type)
    (; Y, Deltat, eps, alpha, DeltatRW) = sc

    if all(iszero, alpha)
        # ── Linear case: fast vectorised analytic discriminant ──────────────
        if application == "RP-RW"
            q = (DeltatRW ./ n0).^2 .+ rt .* (l32 .* l21 .* eps ./ DeltaW).^2 ./ (n0 * mu)
            r = -Y .* DeltatRW.^2 ./ n0^2
        elseif application == "RP-IW"
            q = (Deltat ./ n0).^2 .+ (l21 .* eps).^2 ./ mu
            r = -Y .* Deltat.^2 ./ n0^2
        else
            error("calculate_bifurcation_bounds not implemented for application: $(application)")
        end
        a = -(Y.^2) ./ 3.0 .+ q
        b = 2.0 .* (-Y).^3 ./ 27.0 .- q .* (-Y) ./ 3.0 .+ r
        return reshape(b.^2 ./ 4 .+ a.^3 ./ 27, grid_size, grid_size)

    else
        # ── NL saturation case: numerical root-counting per grid point ──────
        # ψ search range: avoid ψ=0 (trivial) and cap at a physically large value
        psi_lo = 1e-6
        psi_hi = 50.0
        n = length(Y)
        indicator = Vector{Float64}(undef, n)
        for i in eachindex(Y)
            indicator[i] = _nl_bifurcation_indicator(
                application, Y[i], Deltat[i], eps[i], alpha[i], DeltatRW[i],
                n0, mu, rt, l21, l32, DeltaW, psi_lo, psi_hi)
        end
        return reshape(indicator, grid_size, grid_size)
    end
end

"""
    _nl_bifurcation_indicator(application, C1, Δ, ε, α, ΔRW, n0, μ, rt, l21, l32, DW, ψ_lo, ψ_hi)

Return -1.0 if the NL steady-state equation H(ψ) has ≥ 2 positive real roots
(multiple equilibria → locking possible), +1.0 otherwise.

Steady-state residuals (derived by setting dψ/dt = 0 in the normalised ODE):

  RP-IW:  H(ψ) = ψ²·[Δ²·(1+αψ)²·(μ+n₀·rₜ·ψ²)² + (n₀·μ·C₁)²] - (l₂₁·ε)²·(μ+n₀·rₜ·ψ²)²

  RP-RW:  G(ψ) = ψ²·(μ+n₀·rₜ·ψ²)²·(ΔRW+Δ·α·ψ)² - (l₃₂·l₂₁·ε/DW)²·(μ+n₀·rₜ·ψ²)² + (n₀·μ·C₁)²·ψ²
"""
function _nl_bifurcation_indicator(application::String,
                                    C1::Float64, Δ::Float64, ε::Float64,
                                    α::Float64, ΔRW::Float64,
                                    n0::Int, μ::Float64, rt::Float64,
                                    l21::Float64, l32::Float64, DW::Float64,
                                    ψ_lo::Float64, ψ_hi::Float64)
    Q = n0 * μ * C1    # torque coupling term
    r = n0 * rt        # combined factor in (μ + r·ψ²)

    H = if application == "RP-IW"
        P = l21 * ε
        ψ -> ψ^2 * (Δ^2 * (1 + α*ψ)^2 * (μ + r*ψ^2)^2 + Q^2) -
             P^2 * (μ + r*ψ^2)^2
    elseif application == "RP-RW"
        Prw = l32 * l21 * ε / DW
        ψ -> ψ^2 * (μ + r*ψ^2)^2 * (ΔRW + Δ*α*ψ)^2 -
             Prw^2 * (μ + r*ψ^2)^2 +
             Q^2 * ψ^2
    else
        error("_nl_bifurcation_indicator not implemented for application: $(application)")
    end

    # find_zeros throws DomainError when no sign changes are found (function is
    # one-signed across the range → single equilibrium → no locking)
    roots = try
        Roots.find_zeros(H, ψ_lo, ψ_hi)
    catch e
        e isa DomainError ? Float64[] : rethrow(e)
    end
    return length(roots) >= 2 ? -1.0 : 1.0
end


# ─────────────────────────────────────────────────────────────────────────────
#  Single-case time-dependent simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate_one_case(ode_params, application, n_tor, control_type, source_torque,
                       t_final, time_steps) → (sol, norm_t)

Solve a single time-dependent case using `ode_params`'s "harvested" Control2
value (`stability_index`, `error_field`, or `saturation_param`, depending on
`control_type`), and return both the raw `ODESolution` (for physical-unit
plotting) and the normalized time traces `norm_t` (n_times × n_states).
"""
function simulate_one_case(ode_params::ODEparams, application::String, n_tor::Int, EFphase::Float64, control_type::Symbol,
                            source_torque::Real, t_final::Real, time_steps::Int)
    control1 = Float64(source_torque)

    # harvest what's already under the hood to set these inputs
    if control_type == :EF
        control2 = ode_params.error_field
    elseif control_type == :LinStab
        control2 = ode_params.stability_index - 0.5 # small adjustment to move away from marginality
    elseif control_type == :NLsaturation
        control2 = ode_params.saturation_param
    else
        @info "Control scenario NOT set, guessing the value of control2"
        control2 = 0.5
    end

    sol = solve_ODEs(ode_params, application, n_tor, EFphase, control_type, t_final, time_steps, control1, control2; full_output=true)

    norm_t = reduce(vcat, (normalize_ode_results(u, ode_params, control2, control1, control_type)'
                            for u in sol.u))
    println("final normalized solution = ", norm_t[end, :])

    return sol, norm_t
end
