function gp = config(gp, pop_size, num_gen, tournament_size, timeout, X_train, y_train, X_pred)
    % config - Configuration for genetic programming with Python integration.

    try
        % Convert input data to MATLAB arrays (assume they are numpy arrays or lists)
        x_train_matrix = double(X_train);

        % If X_pred is provided, convert it to double, otherwise leave it empty
        if ~isempty(X_pred)
            x_pred_matrix = double(X_pred);
        else
            x_pred_matrix = [];
        end

        % Ensure y_train_matrix is a column vector
        y_train_matrix = double(y_train);
        if isrow(y_train_matrix)
            y_train_matrix = y_train_matrix';
        end

        % Display shapes of the matrices
        disp('Data fetched from Python successfully.');
        disp('Shape of x_train_matrix:');
        disp(size(x_train_matrix));
        disp('Shape of y_train_matrix (after possible conversion):');
        disp(size(y_train_matrix));

        if ~isempty(x_pred_matrix)
            disp('Shape of x_pred_matrix (new prediction data):');
            disp(size(x_pred_matrix));
        end

    catch ME
        disp('Failed to fetch data from Python:');
        disp(ME.message);
        return;
    end

    % Assign pre-processed data to the gp structure
    gp.userdata.xtrain = x_train_matrix;
    gp.userdata.ytrain = y_train_matrix;
    
    if ~isempty(x_pred_matrix)
        gp.userdata.xpred = x_pred_matrix;  % Store X_pred if available
    else
        gp.userdata.xpred = [];  % Empty if no new prediction data provided
    end

    % GP configuration parameters
    gp.runcontrol.timeout = timeout;         % Time limit for the GP run
    gp.runcontrol.pop_size = pop_size;       % Population size for GP
    gp.runcontrol.num_gen = num_gen;         % Number of generations for GP
    gp.selection.tournament.size = tournament_size; % Tournament selection size

    % Configure gene and function node settings
    gp.genes.max_genes = 11; % Maximum number of genes
    gp.nodes.functions.name = {'times', 'minus', 'plus', 'sqrt', 'square', 'sin', 'cos', 'exp', 'add3', 'mult3'}; % Function names

    disp('GP configuration completed.');
end