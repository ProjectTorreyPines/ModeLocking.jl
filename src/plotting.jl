# ─────────────────────────────────────────────────────────────────────────────
#  Single-case time-dependent traces
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_time_traces(norm_t, t) → Plot

Plot the time-dependent, normalized tearing-mode amplitude (TM, ψ_tn) and
resistive-wall-mode amplitude (RWM, ψ_wn — only for the RP-RW system) on the
left y-axis, and the normalized rotation frequency (Ω_tN) vs time on a
secondary (right) y-axis, all on a single labeled plot.

`norm_t` is the (n_times × n_states) matrix of per-timestep normalized
states (as produced by `normalize_ode_results` applied to each `sol.u[i]`),
and `t` is the corresponding vector of times (`sol.t`).
"""
function plot_time_traces(norm_t::AbstractMatrix{<:Real}, t::AbstractVector{<:Real};
                          t0::Real=NaN)
    psi_tN = norm_t[:, 1]
    Om_tN  = norm_t[:, 3]

    t_plot, xlabel_t = if !isnan(t0)
        t .* t0 .* 1e3, "time (ms)"
    else
        t, "time"
    end

    plt = plot(t_plot, psi_tN; xlabel=xlabel_t, ylabel="Normalized ψ", label="ψ_tN (TM)",
               lw=4, color=:steelblue, title="Time-dependent traces",
               legend=:topleft, legendfontsize=12, ylims=(0, 1.05))

    if size(norm_t, 2) == 5 # RP-RW layout includes the resistive-wall-mode state
        psi_wN = norm_t[:, 4]
        plot!(plt, t_plot, psi_wN; label="ψ_wN (RWM)", lw=4, color=:darkorange)
    end

    # Ω_tN on its own scale on the right y-axis; cap at 1+ε since Ω_tN ∈ [0,1]
    ax2 = twinx(plt)
    plot!(ax2, t_plot, Om_tN; ylabel="Ω_tN", label="Ω_tN", lw=4,
          color=:firebrick, linestyle=:dash, legend=:bottomright, legendfontsize=12)
    ylims!(ax2, 0, 1.05)

    return plt
end


"""
    plot_time_traces(sol, norm_vec) → Plot

Same layout as `plot_time_traces(norm_t, t)`, but using the *raw*
(un-normalized) ODE state `sol.u` converted to physical units:

- Magnetic amplitudes (ψ, ψ_W) are normalized in the ODE to `psi0 = b0*r0`
  (units T·m). Converting to a perturbed-field amplitude divides by the
  gradient length scale `r0`, so `δB[T] = ψ * psi0 / r0 = ψ * b0`, i.e.
  `δB[Gauss] = ψ * b0 * 1e4`.
- The rotation state Ω is normalized such that `Ω_real[rad/s] = Ω / t0`
  (from dθ/dt_ode = -m0*Ω with t_ode = t_real/t0). Converting to 2π*kHz:
  `f[2π*kHz] = Ω_real/(2π*1000) = Ω * 1e-3/(t0)`.

`norm_vec = [b0, t0, r0]`.
"""
function plot_time_traces(sol::ODESolution, norm_vec::AbstractVector{<:Real})

    b0, t0, r0 = norm_vec[1], norm_vec[2], norm_vec[3]
    # ψ_ode is normalized to b0 (errF = EF_Gauss*1e-4/b0 → dimensionless).
    # δB[T] = ψ_ode * b0  →  δB[Gauss] = ψ_ode * b0 * 1e4
    mag_factor  = b0 * 1.e4
    freq_factor = 1.e-3 / t0   # Ω (dimensionless) → 2π*kHz

    t_ms    = sol.t .* t0 .* 1e3   # dimensionless ODE time → ms
    psi_G   = [u[1] for u in sol.u] .* mag_factor
    Om_kHz  = [u[3] for u in sol.u] .* freq_factor

    plt = plot(t_ms, psi_G; xlabel="time (ms)", ylabel="δB (Gauss)", label="ψ (TM)",
               lw=4, color=:steelblue, title="Time-dependent traces (physical units)",
               legend=:topleft, legendfontsize=12)

    if length(sol.u[1]) == 5 # RP-RW layout includes the resistive-wall-mode state
        psiW_G = [u[4] for u in sol.u] .* mag_factor
        plot!(plt, t_ms, psiW_G; label="ψ_W (RWM)", lw=4, color=:darkorange)
    end

    # Ω in kHz on its own scale on the right y-axis
    ax2 = twinx(plt)
    plot!(ax2, t_ms, Om_kHz; ylabel="Ω (kHz)", label="Ω", lw=4,
          color=:firebrick, linestyle=:dash, legend=:bottomright, legendfontsize=12)
    ylims!(ax2, 0, maximum(Om_kHz) * 1.05)

    return plt
end


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────
"""
    plot_sols(results, ode_params, grid_size, control_type) → (fig1, fig2, fig3)

Calls `plot_scatter`, `plot_phase_diagrams`, and (when a trained NN model is
available) `plot_probability`.  Returns all three handles; fig3 is `nothing`
when no model has been trained yet.
"""
function plot_sols(results::LockingResults, ode_params::ODEparams, grid_size::Int, control_type::Symbol;
                   b0::Float64=NaN, t0::Float64=NaN, m_pol::Float64=NaN, orientation::Symbol=:portrait,
                   shot_label::String="")
    fig1 = plot_scatter(results; orientation, shot_label); display(fig1)
    fig2 = plot_phase_diagrams(results, ode_params, grid_size, control_type; b0, t0, m_pol, orientation, shot_label); display(fig2)
    fig3 = (results.prob !== nothing) ?
           (p = plot_probability(results, ode_params, control_type; b0, t0, m_pol, shot_label); display(p); p) : nothing
    return fig1, fig2, fig3
end


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — multi-panel scatter of state variables
# ─────────────────────────────────────────────────────────────────────────────
"""
    plot_scatter(results; orientation=:portrait) → Figure 1

Multi-panel scatter of raw and normalised ODE state variables.

RP-RW (5-state):
  Portrait (3×2, default): left col = raw, right col = normalised
    (a) raw ψ_t vs Ω_t   | (b) ψ_tn vs Ω_n (coloured by k-means class)
    (c) raw ψ_w vs Ω_t   | (d) ψ_wn vs Ω_n (coloured)
    (e) ψ_tn vs ψ_wn     | (f) θ_t vs θ_w
  Landscape (2×3): top row = raw panels (a,c,e); bottom row = normalised (b,d,f)

RP-IW (3-state):
  Portrait (2×1): (a) on top, (b) below
  Landscape (1×2): (a) | (b) side by side
"""
function plot_scatter(results::LockingResults; orientation::Symbol=:portrait, shot_label::String="")
    r = results

    is_RPRW = size(r.norm_sols, 2) == 5

    psi_t   = r.ode_sols[:, 1]
    omega_t = r.ode_sols[:, 3]
    psi_tn  = r.norm_sols[:, 1]
    omega_n = r.norm_sols[:, 3]

    idx_U = findall(r.locking_labels .== 1)   # unlocked
    idx_L = findall(r.locking_labels .== 2)   # locked

    # ── (a) raw ψ_t vs Ω_t ──────────────────────────────────────────────────
    p_a = scatter(psi_t, omega_t;
        xlabel          = "ψ_t",
        ylabel          = "Ω_t",
        label           = false,
        alpha           = 0.3,
        markersize      = 3,
        markerstrokewidth = 0,
        left_margin     = 8Plots.mm,
        grid            = true,
    )
    xra = extrema(psi_t); yra = extrema(omega_t)
    annotate!(p_a, xra[1] + 0.02*(xra[2]-xra[1]),
                   yra[1] + 0.04*(yra[2]-yra[1]),
                   Plots.text("(a)", 12, :left))

    # ── (b) ψ_tn vs Ω_n coloured by class ───────────────────────────────────
    p_b = scatter(psi_tn[idx_U], omega_n[idx_U];
        xlabel          = "ψ_tn",
        ylabel          = "Ω_n",
        label           = "Unlocked",
        color           = :steelblue,
        alpha           = 0.3,
        markershape     = :circle,
        markersize      = 4,
        markerstrokewidth = 0,
        left_margin     = 8Plots.mm,
        grid            = true,
    )
    scatter!(p_b, psi_tn[idx_L], omega_n[idx_L];
        label       = "Locked",
        color       = :red,
        alpha       = 0.5,
        markershape = :xcross,
        markersize  = 5,
    )
    xlims!(p_b, -0.02, 1.06)
    ylims!(p_b, -0.02, 1.06)
    annotate!(p_b, 0.02, 0.02, Plots.text("(b)", 12, :left))

    if is_RPRW
        psi_w   = r.ode_sols[:, 4]
        psi_wn  = r.norm_sols[:, 4]
        theta_t = r.norm_sols[:, 2]
        theta_w = r.norm_sols[:, 5]

        # ── (c) raw ψ_w vs Ω_t ──────────────────────────────────────────────
        p_c = scatter(psi_w, omega_t;
            xlabel          = "ψ_w",
            ylabel          = "Ω_t",
            label           = false,
            color           = :steelblue,
            alpha           = 0.3,
            markersize      = 3,
            markerstrokewidth = 0,
            left_margin     = 8Plots.mm,
            grid            = true,
        )
        xrc = extrema(psi_w); yrc = extrema(omega_t)
        annotate!(p_c, xrc[1] + 0.02*(xrc[2]-xrc[1]),
                       yrc[1] + 0.04*(yrc[2]-yrc[1]),
                       Plots.text("(c)", 12, :left))

        # ── (d) ψ_wn vs Ω_n coloured by class ──────────────────────────────
        p_d = scatter(psi_wn[idx_U], omega_n[idx_U];
            xlabel          = "ψ_wn",
            ylabel          = "Ω_n",
            label           = "Unlocked",
            color           = :steelblue,
            alpha           = 0.3,
            markershape     = :circle,
            markersize      = 4,
            markerstrokewidth = 0,
            left_margin     = 8Plots.mm,
            grid            = true,
        )
        scatter!(p_d, psi_wn[idx_L], omega_n[idx_L];
            label       = "Locked",
            color       = :red,
            alpha       = 0.5,
            markershape = :xcross,
            markersize  = 5,
        )
        xlims!(p_d, -0.02, 1.06)
        ylims!(p_d, -0.02, 1.06)
        annotate!(p_d, 0.02, 0.02, Plots.text("(d)", 12, :left))

        # ── (e) ψ_tn vs ψ_wn ────────────────────────────────────────────────
        p_e = scatter(psi_tn, psi_wn;
            xlabel          = "ψ_tn",
            ylabel          = "ψ_wn",
            label           = false,
            color           = :steelblue,
            alpha           = 0.3,
            markersize      = 3,
            markerstrokewidth = 0,
            left_margin     = 8Plots.mm,
            grid            = true,
        )
        xre = extrema(psi_tn); yre = extrema(psi_wn)
        annotate!(p_e, xre[1] + 0.02*(xre[2]-xre[1]),
                       yre[1] + 0.04*(yre[2]-yre[1]),
                       Plots.text("(e)", 12, :left))

        # ── (f) θ_t vs θ_w ──────────────────────────────────────────────────
        p_f = scatter(theta_t, theta_w;
            xlabel          = "θ_t (rad)",
            ylabel          = "θ_w (rad)",
            label           = false,
            color           = :steelblue,
            alpha           = 0.3,
            markersize      = 3,
            markerstrokewidth = 0,
            left_margin     = 8Plots.mm,
            grid            = true,
        )
        xrf = extrema(theta_t); yrf = extrema(theta_w)
        annotate!(p_f, xrf[1] + 0.02*(xrf[2]-xrf[1]),
                       yrf[1] + 0.04*(yrf[2]-yrf[1]),
                       Plots.text("(f)", 12, :left))

        if orientation == :landscape
            # 2×3: top row = raw (a,c,e), bottom row = normalised (b,d,f)
            plt = plot(p_a, p_c, p_e, p_b, p_d, p_f; layout=(2, 3), size=(1500, 700))
        else
            # 3×2 portrait: left col = raw, right col = normalised
            plt = plot(p_a, p_b, p_c, p_d, p_e, p_f; layout=(3, 2), size=(900, 1050))
        end
    else
        if orientation == :landscape
            plt = plot(p_a, p_b; layout=(1, 2), size=(1200, 400))
        else
            plt = plot(p_a, p_b; layout=(2, 1), size=(600, 750))
        end
    end

    if !isempty(shot_label)
        plot!(plt; plot_title=shot_label, plot_titlefontsize=16)
    end

    return plt
end


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — phase diagrams (pcolor of Ω_n and ψ_tn over control space)
# ─────────────────────────────────────────────────────────────────────────────
"""
    plot_phase_diagrams(results, ode_params, grid_size, control_type) → Figure 2

Two stacked pcolor panels of normalised solutions over the (C2, C1) control space.
  (a) Ω_n — normalised rotation      (RdBu colormap)
  (b) ψ_tn — normalised TM amplitude (RdBu_r colormap)
Analytic bifurcation boundary (D = 0) overlaid in black when NL saturation is off.
"""
function plot_phase_diagrams(results::LockingResults, ode_params::ODEparams, grid_size::Int, control_type::Symbol;
                             b0::Float64=NaN, t0::Float64=NaN, m_pol::Float64=NaN, orientation::Symbol=:portrait,
                             shot_label::String="")
    r  = results
    gs = grid_size
    x  = unique(ode_params.Control2)   # Control2 values  (x-axis)
    y  = unique(ode_params.Control1)   # Control1/Ω0 values (y-axis)

    OmN_grid   = reshape(r.norm_sols[:, 3], gs, gs)   # Ω_n   (rows=C1, cols=C2)
    PsiTN_grid = reshape(r.norm_sols[:, 1], gs, gs)   # ψ_tn  (rows=C1, cols=C2)
    bb         = r.bifurcation_bounds                  # local; may be reversed below

    # Convert axes to physical units when normalization scales are available
    if control_type == :EF && !isnan(b0) && !isnan(m_pol)
        x           = x .* (m_pol / ode_params.control_surf) .* b0 .* 1e4
        xlabel_ctrl = "Error Field (Gauss)"
    elseif control_type == :NLsaturation
        # 1/α inverts sort order — reverse x and all column-ordered matrices
        x           = reverse(1.0 ./ x)
        OmN_grid    = reverse(OmN_grid,   dims=2)
        PsiTN_grid  = reverse(PsiTN_grid, dims=2)
        bb          = bb === nothing ? nothing : reverse(bb, dims=2)
        xlabel_ctrl = "1/α"
    else
        xlabel_ctrl = _ctrl_xlabel(control_type)
    end
    if !isnan(t0)
        y          = y ./ (t0 * 1e3)
        ylabel_str = "Ω₀ (2π*kHz)"
    else
        ylabel_str = "Ω₀"
    end

    xr = extrema(x); yr = extrema(y)
    bot_margin = orientation == :landscape ? 12Plots.mm : 4Plots.mm

    # ── (a) Ω_n ─────────────────────────────────────────────────────────────
    p_a = heatmap(x, y, OmN_grid;
        xlabel         = xlabel_ctrl,
        ylabel         = ylabel_str,
        title          = "Ω_n",
        color          = cgrad(:RdBu),
        colorbar_title = "Ω_n",
        clims          = (0.0, 1.0),
        left_margin    = 8Plots.mm,
        bottom_margin  = bot_margin,
        guidefontsize  = 14,
        tickfontsize   = 12,
        titlefontsize  = 18,
    )
    _overlay_bifurcation!(p_a, x, y, bb)
    annotate!(p_a, xr[1]+0.05*(xr[2]-xr[1]), yr[1]+0.85*(yr[2]-yr[1]),
              Plots.text("UNLOCKED", 18, :white, :left))
    annotate!(p_a, xr[1]+0.65*(xr[2]-xr[1]), yr[1]+0.05*(yr[2]-yr[1]),
              Plots.text("LOCKED",   18, :white, :left))
    annotate!(p_a, xr[1]+0.01*(xr[2]-xr[1]), yr[1]+0.05*(yr[2]-yr[1]),
              Plots.text("(a)", 15, :white, :left))

    # ── (b) ψ_tn ────────────────────────────────────────────────────────────
    p_b = heatmap(x, y, PsiTN_grid;
        xlabel         = xlabel_ctrl,
        ylabel         = ylabel_str,
        title          = "ψ_tn",
        color          = cgrad(:RdBu, rev=true),
        colorbar_title = "ψ_tn",
        clims          = (0.0, 1.0),
        left_margin    = 8Plots.mm,
        bottom_margin  = bot_margin,
        guidefontsize  = 14,
        tickfontsize   = 12,
        titlefontsize  = 18,
    )
    _overlay_bifurcation!(p_b, x, y, bb)
    annotate!(p_b, xr[1]+0.05*(xr[2]-xr[1]), yr[1]+0.85*(yr[2]-yr[1]),
              Plots.text("UNLOCKED", 18, :white, :left))
    annotate!(p_b, xr[1]+0.65*(xr[2]-xr[1]), yr[1]+0.05*(yr[2]-yr[1]),
              Plots.text("LOCKED",   18, :white, :left))
    annotate!(p_b, xr[1]+0.01*(xr[2]-xr[1]), yr[1]+0.05*(yr[2]-yr[1]),
              Plots.text("(b)", 15, :white, :left))

    plt = orientation == :landscape ?
        plot(p_a, p_b; layout=(1, 2), size=(1400, 550)) :
        plot(p_a, p_b; layout=(2, 1), size=(650, 1100))

    if !isempty(shot_label)
        plot!(plt; plot_title=shot_label, plot_titlefontsize=16)
    end

    return plt
end


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — NN locking probability
# ─────────────────────────────────────────────────────────────────────────────
"""
    plot_probability(results, ode_params, control_type) → Figure 3

Contourf of NN locking probability P(locked) over the (C2, C1) control space.
  • dashed black  : P = 0.5 decision boundary
  • dashed yellow : analytic bifurcation boundary (D = 0), when available
"""
function plot_probability(results::LockingResults, ode_params::ODEparams, control_type::Symbol;
                          b0::Float64=NaN, t0::Float64=NaN, m_pol::Float64=NaN,
                          shot_label::String="")
    r = results
    r.prob === nothing && error("No trained NN model — run train_locking_nn first")

    x_ctrl = unique(ode_params.Control2)   # original Control2 — always used for NN eval
    x      = copy(x_ctrl)                  # display axis — may be transformed below
    y      = unique(ode_params.Control1)

    # Convert axes to physical units when normalization scales are available
    if control_type == :EF && !isnan(b0) && !isnan(m_pol)
        x           = x .* (m_pol / ode_params.control_surf) .* b0 .* 1e4
        xlabel_ctrl = "Error Field (Gauss)"
    elseif control_type == :NLsaturation
        # 1/α inverts sort order — reverse display axis and grid columns to match
        x           = reverse(1.0 ./ x)
        xlabel_ctrl = "1/α"
    else
        xlabel_ctrl = _ctrl_xlabel(control_type)
    end
    ylabel_str = !isnan(t0) ? "Ω₀ (2π*kHz)" : "Ω₀"
    if !isnan(t0)
        y = y ./ (t0 * 1e3)
    end

    prob_grid = [r.prob(c1, c2) for c1 in y, c2 in x_ctrl]
    bb = r.bifurcation_bounds
    if control_type == :NLsaturation
        prob_grid = reverse(prob_grid, dims=2)
        bb        = bb === nothing ? nothing : reverse(bb, dims=2)
    end
    xr = extrema(x); yr = extrema(y)

    plt = contourf(x, y, prob_grid;
        xlabel         = xlabel_ctrl,
        ylabel         = ylabel_str,
        title          = "Locking probability P(locked) — NN",
        colorbar_title = "P(locked)",
        clims          = (0.0, 1.0),
        levels         = 20,
        color          = cgrad(:RdBu, rev=true),
        colorbar       = true,
        left_margin    = 8Plots.mm,
        right_margin   = 4Plots.mm,
        guidefontsize  = 14,
        tickfontsize   = 12,
        titlefontsize  = 16,
    )
    contour!(plt, x, y, prob_grid;
        levels    = [0.5],
        linecolor = :black,
        linestyle = :dash,
        linewidth = 2.5,
        label     = "P = 0.5",
    )
    _overlay_bifurcation!(plt, x, y, bb; color=:yellow, style=:dash)

    annotate!(plt, xr[1]+0.05*(xr[2]-xr[1]), yr[1]+0.85*(yr[2]-yr[1]),
              Plots.text("UNLOCKED", 18, :white, :left))
    annotate!(plt, xr[1]+0.70*(xr[2]-xr[1]), yr[1]+0.05*(yr[2]-yr[1]),
              Plots.text("LOCKED",   18, :white, :left))

    if !isempty(shot_label)
        annotate!(plt, xr[1]+0.02*(xr[2]-xr[1]), yr[2]-0.05*(yr[2]-yr[1]),
                  Plots.text(shot_label, 14, :white, :left))
    end

    return plt
end


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

"Return the x-axis label for the swept control parameter C2"
function _ctrl_xlabel(control_type::Symbol)
    control_type == :EF           ? "Error Field"         :
    control_type == :LinStab      ? "Linear Stability Δ′" :
    control_type == :NLsaturation ? "NL Saturation α"     : "Control 2"
end

"""
    _zero_isoline(x, y, z) → (xs, ys)

Compute the D=0 isoline of a 2-D scalar field `z` defined on grid `(x, y)`
by linear interpolation along cell edges (simplified marching squares).
Returns flat vectors suitable for `plot!`; NaN separates disjoint segments.
`z` must be (length(y) × length(x)) — the same convention as Plots.jl heatmap.
"""
function _zero_isoline(x::AbstractVector, y::AbstractVector, z::AbstractMatrix)
    xs = Float64[]
    ys = Float64[]
    nx, ny = length(x), length(y)   # x → columns, y → rows

    for i in 1:ny-1, j in 1:nx-1
        pts = NTuple{2,Float64}[]

        # bottom edge: row i,   col j → j+1
        v1, v2 = z[i,j], z[i,j+1]
        if v1 * v2 < 0
            t = v1 / (v1 - v2)
            push!(pts, (x[j] + t*(x[j+1]-x[j]), y[i]))
        end
        # top edge:    row i+1, col j → j+1
        v1, v2 = z[i+1,j], z[i+1,j+1]
        if v1 * v2 < 0
            t = v1 / (v1 - v2)
            push!(pts, (x[j] + t*(x[j+1]-x[j]), y[i+1]))
        end
        # left edge:   col j,   row i → i+1
        v1, v2 = z[i,j], z[i+1,j]
        if v1 * v2 < 0
            t = v1 / (v1 - v2)
            push!(pts, (x[j], y[i] + t*(y[i+1]-y[i])))
        end
        # right edge:  col j+1, row i → i+1
        v1, v2 = z[i,j+1], z[i+1,j+1]
        if v1 * v2 < 0
            t = v1 / (v1 - v2)
            push!(pts, (x[j+1], y[i] + t*(y[i+1]-y[i])))
        end

        if length(pts) >= 2
            push!(xs, pts[1][1], pts[2][1], NaN)
            push!(ys, pts[1][2], pts[2][2], NaN)
        end
    end
    return xs, ys
end

"""
Overlay analytic bifurcation boundary (D=0 isoline) — no-op when bb is nothing.

Drawn as a plain `plot!` line series from manually-computed crossing points,
rather than a `contour!` series: heatmap + contour! on one subplot share a
single color/z-scale in GR, and `bb`'s native range (e.g. up to ~700) versus
the heatmap's `[0,1]` range makes that shared scale unworkable either way
(heatmap goes flat, or the D=0 level gets clipped away). A line series has no
z/colormap at all, so it cannot disturb the heatmap's color scale.
"""
function _overlay_bifurcation!(p, x, y, bb; color=:black, style=:solid)
    bb === nothing && return
    xs, ys = _zero_isoline(x, y, bb)
    isempty(xs) && return
    plot!(p, xs, ys; linecolor=color, linestyle=style, linewidth=2.5, label=false)
end


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

function make_contour(X::AbstractArray, Y::AbstractArray, Z::AbstractMatrix)
    # Determine target shape
    m, n = size(Z)

    # If C1 and C2 are vectors, try to reshape them
    if ndims(X) == 1
        X = unique(X)
    end
    if ndims(Y) == 1
        Y = unique(Y)
    end

    plt = plot()
    plt = heatmap!(plt, X,Y, Z)#; linewidth=2)
    display(plt)   # explicitly display
    return plt
end

function make_contour(X::AbstractArray, Y::AbstractArray, Z::AbstractMatrix, levels::Vector{Float64}, control_type)
    lblsz = 16

    # Ensure unique 1D grids
    if ndims(X) == 1
        X = unique(X)
    end
    if ndims(Y) == 1
        Y = unique(Y)
    end

    # Axis label
    xlabel = control_type == :EF          ? "Error Field" :
             control_type == :LinStab     ? "Linear Stability" :
             control_type == :NLsaturation ? "NL saturation" : "Control1"

    # Contour plot
    plt = contour(X, Y, Z; levels=levels, linewidth=2,
                  xlabel=xlabel, ylabel="Normalized Torque",
                  clabel=false)

    # Compute axis ranges for relative placement
    xmin, xmax = extrema(X)
    ymin, ymax = extrema(Y)

    annotate!(xmin + 0.5*(xmax-xmin), ymin + 0.85*(ymax-ymin), text("UNLOCKED", lblsz))
    annotate!(xmin + 0.80*(xmax-xmin), ymin + 0.02*(ymax-ymin), text("LOCKED", lblsz))

    if any(Z .< 0)
        ind = argmin(Z)
        i, j = Tuple(ind)   # row, col indices
        xloc = 0.9*X[j]
        yloc = 0.9*Y[i]
        annotate!(xloc, yloc, text("Locking\n(Possible)", lblsz, :red))
    end

    display(plt)
    return plt
end
