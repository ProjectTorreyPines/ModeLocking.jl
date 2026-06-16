using Test
using ModeLocking
using ModeLocking: ODEparams, NNparams, LockingResults, LockingNNModel
using Flux: Chain
using Plots

# A representative set of derived quantities, matching what
# FUSE.calculate_stability_index!/set_phys_params! would compute for the
# default n_mode=1, rat_surface=0.67, res_wall=1.0, control_surf=1.25,
# RPRW_stability_index=-0.5 case.
function _test_ode_params(; control_type=:EF, NL_saturation_ON=false)
    sat = NL_saturation_ON ? 0.1 : 0.0
    return ODEparams(;
        rat_surface = 0.67,
        res_wall    = 1.0,
        control_surf = 1.25,
        mu      = 0.1,
        Inertia = 0.1,
        Taut_Tauw = 1.0,
        l12 = 2.4313, l21 = 3.6285, l32 = 4.4444,
        DeltaW = -7.1841,
        stability_index = -1.7281,
        error_field = 0.5,
        saturation_param = sat,
    )
end

@testset "ModeLocking.jl" begin

    @testset "types" begin
        op = ODEparams()
        @test op isa ODEparams
        @test op.Control1 == Float64[]
        @test op.Control2 == Float64[]

        nnp = NNparams()
        @test nnp.hidden_sizes == [100, 100, 100]
    end

    @testset "resolve_control" begin
        for control_type in (:EF, :LinStab, :NLsaturation)
            op = _test_ode_params(; control_type, NL_saturation_ON=(control_type == :NLsaturation))
            C2 = control_type == :LinStab ? -2.0 : 0.2
            sc = ModeLocking.resolve_control(op, control_type, C2)
            @test all(isfinite, (sc.Deltat, sc.eps, sc.alpha, sc.DeltatRW))
        end

        # :LinStab with an unstable RP-RW mode must error
        op = _test_ode_params(; control_type=:LinStab)
        @test_throws Exception ModeLocking.resolve_control(op, :LinStab, 5.0)

        # Vector resolver over a small grid
        op = _test_ode_params(; control_type=:EF)
        op.Control1 = [1.0, 2.0, 3.0]
        op.Control2 = [0.1, 0.2, 0.3]
        sc = ModeLocking.resolve_control(op, :EF)
        @test length(sc.Deltat) == length(sc.eps) == length(sc.alpha) == length(sc.DeltatRW) == 3
        @test sc.eps == op.Control2
    end

    @testset "solve_ODEs" begin
        for application in ("RP-RW", "RP-IW"), control_type in (:EF, :LinStab, :NLsaturation)
            op = _test_ode_params(; control_type, NL_saturation_ON=(control_type == :NLsaturation))
            C1 = 1.0
            C2 = control_type == :LinStab ? -2.0 : 0.2
            final = ModeLocking.solve_ODEs(op, application, 1, control_type, 10.0, 20, C1, C2)
            @test length(final) == (application == "RP-RW" ? 5 : 3)
            @test all(isfinite, final)
        end
    end

    @testset "normalize_ode_results" begin
        op = _test_ode_params(; control_type=:EF)
        C1, C2 = 1.0, 0.2
        final = ModeLocking.solve_ODEs(op, "RP-RW", 1, :EF, 10.0, 20, C1, C2)

        # single-case dispatch
        norm1 = ModeLocking.normalize_ode_results(final, op, C2, C1, :EF)
        @test length(norm1) == 5
        @test all(isfinite, norm1)

        # batch dispatch
        results = vcat(final', final')
        C2_vec = [C2, C2]
        C1_vec = [C1, C1]
        norm_batch = ModeLocking.normalize_ode_results(results, op, C2_vec, C1_vec, :EF)
        @test size(norm_batch) == (2, 5)
        @test norm_batch[1, :] ≈ norm1
    end

    @testset "solve_grid / solve_and_classify" begin
        op = _test_ode_params(; control_type=:EF)
        grid_size = 3
        op.Control1 = vec(repeat(range(1.0, 3.0, length=grid_size), 1, grid_size))
        op.Control2 = vec(repeat(range(0.05, 0.5, length=grid_size)', grid_size, 1))

        results = ModeLocking.solve_and_classify(op, "RP-RW", 1, :EF, 10.0, 20, false, grid_size)
        @test results isa LockingResults
        @test size(results.ode_sols) == (grid_size^2, 5)
        @test size(results.norm_sols) == (grid_size^2, 5)
        @test length(results.locking_labels) == grid_size^2
        @test all(l ∈ (1, 2) for l in results.locking_labels)
        @test results.prob === nothing
        @test results.bifurcation_bounds !== nothing
        @test size(results.bifurcation_bounds) == (grid_size, grid_size)
        @test all(isfinite, results.bifurcation_bounds)

        # NL saturation suppresses bifurcation bounds
        op2 = _test_ode_params(; control_type=:NLsaturation, NL_saturation_ON=true)
        op2.Control1 = op.Control1
        op2.Control2 = vec(repeat(range(0.05, 0.3, length=grid_size)', grid_size, 1))
        results2 = ModeLocking.solve_and_classify(op2, "RP-RW", 1, :NLsaturation, 10.0, 20, true, grid_size)
        @test results2.bifurcation_bounds === nothing
    end

    @testset "simulate_one_case" begin
        op = _test_ode_params(; control_type=:EF)
        sol, norm_t = ModeLocking.simulate_one_case(op, "RP-RW", 1, :EF, 1.0, 10.0, 20)
        @test size(norm_t, 2) == 5
        @test size(norm_t, 1) == length(sol.u)
        @test all(isfinite, norm_t)
    end

    @testset "classifier" begin
        op = _test_ode_params(; control_type=:EF)
        grid_size = 3
        op.Control1 = vec(repeat(range(1.0, 3.0, length=grid_size), 1, grid_size))
        op.Control2 = vec(repeat(range(0.05, 0.5, length=grid_size)', grid_size, 1))
        results = ModeLocking.solve_and_classify(op, "RP-RW", 1, :EF, 10.0, 20, false, grid_size)

        X, y = ModeLocking.prepare_nn_data(results.locking_labels, op.Control1, op.Control2)
        @test size(X) == (2, grid_size^2)
        @test size(y) == (1, grid_size^2)
        @test all(v -> v in (0.0f0, 1.0f0), y)

        for act in (:tanh, :relu, :sigmoid)
            @test ModeLocking.get_activation(act) isa Function
        end
        @test_throws Exception ModeLocking.get_activation(:bogus)

        nn_params = NNparams(hidden_sizes=[4], n_epochs=5, batch_size=size(X,2))
        chain = ModeLocking.build_locking_nn(nn_params)
        @test chain isa Chain

        prob_model = ModeLocking.train_locking_nn(results, op, nn_params)
        @test prob_model isa LockingNNModel
        @test results.prob === prob_model
        p = prob_model(op.Control1[1], op.Control2[1])
        @test 0.0 <= p <= 1.0

        ft = ModeLocking.transfer_learn_locking_nn(prob_model, X, y; nn_params=NNparams(hidden_sizes=nn_params.hidden_sizes, n_epochs=2, batch_size=size(X,2)))
        @test ft isa LockingNNModel
        pf = ft(op.Control1[1], op.Control2[1])
        @test 0.0 <= pf <= 1.0

        best_params = ModeLocking.tune_locking_nn(results, op; n_trials=1, n_folds=2)
        @test best_params isa NNparams
        @test results.prob isa LockingNNModel
    end

    @testset "persistence" begin
        op = _test_ode_params(; control_type=:EF)
        grid_size = 3
        op.Control1 = vec(repeat(range(1.0, 3.0, length=grid_size), 1, grid_size))
        op.Control2 = vec(repeat(range(0.05, 0.5, length=grid_size)', grid_size, 1))
        results = ModeLocking.solve_and_classify(op, "RP-RW", 1, :EF, 10.0, 20, false, grid_size)

        mktempdir() do dir
            path = ModeLocking.save_ode_results(results, op; dir=dir)
            @test isfile(path)

            loaded, C1, C2 = ModeLocking.load_ode_results(; dir=dir)
            @test loaded isa LockingResults
            @test loaded.ode_sols == results.ode_sols
            @test loaded.norm_sols == results.norm_sols
            @test loaded.locking_labels == results.locking_labels
            @test loaded.bifurcation_bounds == results.bifurcation_bounds
            @test C1 == op.Control1
            @test C2 == op.Control2

            nn_params = NNparams(hidden_sizes=[4], n_epochs=5, batch_size=size(op.Control1,1)^0 + grid_size^2)
            prob_model = ModeLocking.train_locking_nn(results, op, nn_params)
            mpath = ModeLocking.save_locking_nn(prob_model; dir=dir)
            @test isfile(mpath)

            loaded_model = ModeLocking.load_locking_nn(; dir=dir)
            @test loaded_model isa LockingNNModel
            p1 = prob_model(op.Control1[1], op.Control2[1])
            p2 = loaded_model(op.Control1[1], op.Control2[1])
            @test p1 ≈ p2
        end
    end

    @testset "plotting" begin
        op = _test_ode_params(; control_type=:EF)
        grid_size = 3
        op.Control1 = vec(repeat(range(1.0, 3.0, length=grid_size), 1, grid_size))
        op.Control2 = vec(repeat(range(0.05, 0.5, length=grid_size)', grid_size, 1))
        results = ModeLocking.solve_and_classify(op, "RP-RW", 1, :EF, 10.0, 20, false, grid_size)

        sol, norm_t = ModeLocking.simulate_one_case(op, "RP-RW", 1, :EF, 1.0, 10.0, 20)
        @test ModeLocking.plot_time_traces(norm_t, sol.t) isa Plots.Plot
        @test ModeLocking.plot_time_traces(sol, [1.0, 1.0, 1.0]) isa Plots.Plot

        @test ModeLocking.plot_scatter(results) isa Plots.Plot
        @test ModeLocking.plot_phase_diagrams(results, op, grid_size, :EF) isa Plots.Plot

        nn_params = NNparams(hidden_sizes=[4], n_epochs=5, batch_size=grid_size^2)
        ModeLocking.train_locking_nn(results, op, nn_params)
        @test ModeLocking.plot_probability(results, op, :EF) isa Plots.Plot

        @test ModeLocking._ctrl_xlabel(:EF) == "Error Field"
        @test ModeLocking._ctrl_xlabel(:LinStab) == "Linear Stability Δ′"
        @test ModeLocking._ctrl_xlabel(:NLsaturation) == "NL Saturation α"
        @test ModeLocking._ctrl_xlabel(:other) == "Control 2"
    end

end
