function [ normH ] = norm_H3( H_CSI )

% Normalize data to [0,1]. Min-Max Feature scaling
normH = H_CSI - min(H_CSI(:));
normH = normH ./ max((normH(:)));

end

