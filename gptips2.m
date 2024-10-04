function [output] = gptips2(X_train, y_train, X_pred, pop_size, num_gen, tournament_size, timeout)
    addpath(genpath('C:\Users\mohammad\Desktop\gptips\gptips2'));
    
    % Initialize the gp structure
    gp = struct();  % Initialize gp as an empty structure

    % Convert parameters to double to avoid type conflicts
    pop_size = double(pop_size);
    num_gen = double(num_gen);
    tournament_size = double(tournament_size);
    timeout = double(timeout);

    % Set up GP configuration for training
    % This is where we pass the training data and GP settings to the config function
    gp = rungp(@(gp) config(gp, pop_size, num_gen, tournament_size, timeout, X_train, y_train, X_pred));

    % Check if the gp structure was correctly set up
    if isempty(gp)
        error('GP configuration failed.');
    end

    % Use the provided model to make predictions
    gp_func = gpmodel2func(gp, 'best');  % Extract best GP model function

    % Extract function argument info (e.g., 'x1', 'x2', etc.)
    func_info = functions(gp_func);
    func_string = func_info.function;

    % Get the argument variables (input features)
    startIdx = strfind(func_string, '@(') + 2;
    endIdx = strfind(func_string, ')') - 1;
    args_string = func_string(startIdx:endIdx);
    args_list = strsplit(args_string, ',');

    numeric_indices = zeros(1, length(args_list));

    % Extract numeric indices from arguments (e.g., 'x2' -> 2)
    for i = 1:length(args_list)
        arg_name = args_list{i};
        numeric_part = arg_name(2:end);  % Get the numeric part of 'xN'
        numeric_indices(i) = str2double(numeric_part);
    end

    % Perform predictions for training data
    args_values_train = X_train(:, numeric_indices);
    args_cell_train = num2cell(args_values_train, 1);  % Convert train data to cell array
    y_pred_train = gp_func(args_cell_train{:});  % Predict on training data

    % Perform predictions for the new prediction data (X_pred)
    if ~isempty(X_pred)
        args_values_pred = X_pred(:, numeric_indices);
        args_cell_pred = num2cell(args_values_pred, 1);  % Convert prediction data to cell array
        y_pred_pred = gp_func(args_cell_pred{:});  % Predict on new data
    else
        y_pred_pred = [];
    end

    % Return the predictions for training and new data
    output.y_pred_train = y_pred_train;
    output.y_pred_pred = y_pred_pred;
end
