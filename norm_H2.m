function [ normH ] = norm_H2( H_CSI )

% Normalize data to [0.5,1]. Min-Max Feature scaling
H_CSI = abs(H_CSI);
normH = H_CSI - min(H_CSI(:));
normH = normH ./ max((normH(:)));
normH = normH + 1;
normH = normH/2;

end

