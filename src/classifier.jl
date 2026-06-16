# ─────────────────────────────────────────────────────────────────────────────
#  Neural-network classifier (Task 1 / Task 2)
# ─────────────────────────────────────────────────────────────────────────────

"""
    prepare_nn_data(locking_labels, control1, control2) → (X, y)

Prepare (X, y) training data from ODE results.
  X : (2 × N) Float32 — raw [C1, C2] values
  y : (1 × N) Float32 — k-means labels shifted from {1,2} → {0,1}
"""
function prepare_nn_data(locking_labels::Vector{Int},
                          control1::Vector{Float64}, control2::Vector{Float64})
    X = Float32.(hcat(control1, control2)')   # (2 × N)
    y = reshape(Float32.(locking_labels .- 1), 1, :)   # (1 × N) Matrix{Float32}, {1,2} → {0,1}
    return X, y
end

"Map activation symbol to the corresponding Flux function"
function get_activation(act::Symbol)
    act == :tanh    ? tanh    :
    act == :relu    ? relu    :
    act == :sigmoid ? sigmoid :
    error("Unknown activation symbol: $act.  Choose :tanh, :relu, or :sigmoid")
end

"Build a Flux Chain from NNparams: 2 inputs → hidden layers → 1 sigmoid output"
function build_locking_nn(nn_params::NNparams)
    act   = get_activation(nn_params.activation)
    sizes = [2; nn_params.hidden_sizes; 1]
    layers = []
    for i in 1:length(sizes)-2
        push!(layers, Dense(sizes[i], sizes[i+1], act))
    end
    push!(layers, Dense(sizes[end-1], 1, sigmoid))
    return Chain(layers...)
end

"""
Train a Flux model on (X, y) with L2 regularisation.
When `nn_params.val_fraction > 0`, a validation split is held out and
early stopping is applied (patience = `nn_params.patience`).
When `val_fraction == 0` (default), all data is used and training runs
for the full `n_epochs` — appropriate for physics-driven classification
problems where the labels are deterministic.
Returns the best model found (or the final model when validation is off).

An optional pre-built `opt_state` can be supplied (e.g. with some layers
frozen via `Flux.freeze!`, as used by `transfer_learn_locking_nn`); if
omitted, a fresh `Adam` optimizer state is created for the whole model.
"""
function _fit_nn_model(model, X::Matrix{Float32}, y::Matrix{Float32}, nn_params::NNparams;
                        opt_state = Flux.setup(Adam(nn_params.learning_rate), model))
    use_val = nn_params.val_fraction > 0.0

    if use_val
        n_val   = max(1, round(Int, nn_params.val_fraction * size(X, 2)))
        n_train = size(X, 2) - n_val
        X_tr = X[:, 1:n_train];     y_tr = y[:, 1:n_train]
        X_va = X[:, n_train+1:end]; y_va = y[:, n_train+1:end]
    else
        X_tr = X;  y_tr = y
    end

    data       = Flux.DataLoader((X_tr, y_tr); batchsize=nn_params.batch_size, shuffle=true)
    wd         = Float32(nn_params.weight_decay)
    best_val   = Inf
    n_wait     = 0
    best_model = model

    for epoch in 1:nn_params.n_epochs
        for (xb, yb) in data
            _, grads = Flux.withgradient(model) do m
                l2 = sum(l -> sum(abs2, l.weight), m.layers)
                Flux.binarycrossentropy(m(xb), yb) + wd * l2
            end
            Flux.update!(opt_state, model, grads[1])
        end

        if use_val
            val = Flux.binarycrossentropy(model(X_va), y_va)
            if val < best_val
                best_val = val;  n_wait = 0;  best_model = deepcopy(model)
            else
                n_wait += 1
                if n_wait >= nn_params.patience
                    @info "  Early stopping at epoch $epoch  best_val=$(round(best_val; digits=4))"
                    break
                end
            end
            epoch % 200 == 0 && @info "  NN epoch $epoch/$(nn_params.n_epochs)  val=$(round(val; digits=4))"
        else
            epoch % 200 == 0 && @info "  NN epoch $epoch/$(nn_params.n_epochs)  loss=$(round(Flux.binarycrossentropy(model(X), y); digits=4))"
        end
    end
    return best_model
end


"""
    train_locking_nn(results::LockingResults, ode_params::ODEparams, nn_params=NNparams()) → LockingNNModel

Train a binary NN classifier (C1, C2) → P(locked) on the k-means labels in
`results`. Updates `results.prob` in-place and returns the trained model.

The returned model is callable: `model(C1, C2)` ∈ [0, 1].
To search for better hyperparameters first, call `tune_locking_nn(results, ode_params)`.
"""
function train_locking_nn(results::LockingResults, ode_params::ODEparams, nn_params::NNparams=NNparams())
    X, y = prepare_nn_data(results.locking_labels, ode_params.Control1, ode_params.Control2)

    @info "Training NN: architecture=$(nn_params.hidden_sizes)  epochs=$(nn_params.n_epochs)"
    model = _fit_nn_model(build_locking_nn(nn_params), X, y, nn_params)
    @info "NN training complete. Final loss=$(round(Flux.binarycrossentropy(model(X), y); digits=4))"

    prob_model = LockingNNModel(model, nn_params)
    results.prob = prob_model
    return prob_model
end


"""
    transfer_learn_locking_nn(base_model, X_new, y_new; nn_params=base_model.nn_params) → LockingNNModel

Fine-tune `base_model` (typically loaded via `load_locking_nn()`) on new
`(X_new, y_new)` data from a different equilibrium/`dd`, freezing every
layer except the last (output) `Dense` layer.

`nn_params` controls the fine-tuning run (learning rate, epochs, etc.) —
pass a separate, typically gentler, `NNparams` than the one used for the
original full training. Returns a new `LockingNNModel`; does not mutate
`base_model`.
"""
function transfer_learn_locking_nn(base_model::LockingNNModel,
                                    X_new::Matrix{Float32}, y_new::Matrix{Float32};
                                    nn_params::NNparams = base_model.nn_params)
    model     = deepcopy(base_model.model)
    opt_state = Flux.setup(Adam(nn_params.learning_rate), model)

    # Freeze every layer except the last (output) Dense layer
    for i in 1:length(model.layers)-1
        Flux.freeze!(opt_state.layers[i])
    end

    @info "Transfer learning: fine-tuning last layer only ($(length(model.layers)) total layers, $(nn_params.n_epochs) epochs)"
    fine_tuned = _fit_nn_model(model, X_new, y_new, nn_params; opt_state=opt_state)
    @info "Transfer learning complete. Final loss=$(round(Flux.binarycrossentropy(fine_tuned(X_new), y_new); digits=4))"

    return LockingNNModel(fine_tuned, nn_params)
end


"""
    tune_locking_nn(results::LockingResults, ode_params::ODEparams; n_trials=20, n_folds=3,
                    rng=GLOBAL_RNG) → NNparams

Random hyperparameter search using k-fold CV on `results`/`ode_params`.
After finding the best configuration, retrains the full model with those
hyperparameters and updates `results.prob` in-place.

Returns the best `NNparams` (useful for Task 2 transfer learning).
"""
function tune_locking_nn(results::LockingResults, ode_params::ODEparams; n_trials::Int=20, n_folds::Int=3,
                          rng::AbstractRNG=Random.GLOBAL_RNG)
    X, y = prepare_nn_data(results.locking_labels, ode_params.Control1, ode_params.Control2)

    # Search space — mirrors Python's RandomizedSearchCV param_dist
    hidden_pool = [[10,10], [10,20,10], [100,100], [200,100,100], [200,100,100,200]]
    lr_pool     = exp10.(range(-4, -2; length=5))   # logspace(-4,-2,5)
    wd_pool     = exp10.(range(-10, -6; length=5))  # logspace(-10,-6,5)
    batch_pool  = [100, 200, 400, 800]
    epoch_pool  = [400, 1000]                        # CV uses half; final uses full

    N         = size(X, 2)
    fold_size = N ÷ n_folds
    best_params = NNparams()
    best_loss   = Inf

    @info "Hyperparameter search: $n_trials trials, $n_folds-fold CV"
    for trial in 1:n_trials
        params = NNparams(
            hidden_sizes  = rand(rng, hidden_pool),
            activation    = :relu,                       # Python fixes relu
            learning_rate = rand(rng, lr_pool),
            n_epochs      = rand(rng, epoch_pool) ÷ 2,  # half epochs for CV speed
            batch_size    = rand(rng, batch_pool),
            weight_decay  = rand(rng, wd_pool),
            patience      = 20,
        )

        # k-fold cross-validation loss
        cv_loss = 0.0
        for k in 1:n_folds
            val_idx   = ((k-1)*fold_size + 1):min(k*fold_size, N)
            train_idx = setdiff(1:N, val_idx)
            m = _fit_nn_model(build_locking_nn(params), X[:, train_idx], y[:, train_idx], params)
            cv_loss += Flux.binarycrossentropy(m(X[:, val_idx]), y[:, val_idx])
        end
        cv_loss /= n_folds

        @info "  Trial $trial/$n_trials  cv_loss=$(round(cv_loss; digits=4))  arch=$(params.hidden_sizes)  lr=$(round(params.learning_rate;sigdigits=2))  wd=$(round(params.weight_decay;sigdigits=2))"
        if cv_loss < best_loss
            best_loss   = cv_loss
            best_params = params
        end
    end

    @info "Best: arch=$(best_params.hidden_sizes)  act=$(best_params.activation)  lr=$(best_params.learning_rate)  CV_loss=$(round(best_loss; digits=4))"

    # Retrain on full data restoring full epoch count (CV used half)
    full_params = NNparams(
        hidden_sizes  = best_params.hidden_sizes,
        activation    = best_params.activation,
        learning_rate = best_params.learning_rate,
        n_epochs      = best_params.n_epochs * 2,
        batch_size    = best_params.batch_size,
        weight_decay  = best_params.weight_decay,
        patience      = best_params.patience,
    )
    @info "Retraining final model ($(full_params.n_epochs) epochs)..."
    train_locking_nn(results, ode_params, full_params)
    return full_params
end
