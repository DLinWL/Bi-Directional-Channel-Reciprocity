%% load the orginal data before normalization
% Each CSI matrix is reshaped into a row vector.
% The real part of CSI matrix corresponds to the first half of this row vector.
% The imaginary part corresponds to the other half.
load('data/mat_indoor5351_bw20Mhz.mat')%load orginal data set
Hur_down(all(Hur_down==0,2),:) = [];
Hur_up(all(Hur_up==0,2),:) = [];
HD_test = Hur_down(70001:100000,:);%test set

%% extract magnitude and phase
bw_size = 1; % bw_size should be 2 when doubling the bandwidth.
csi_matrix_size = 32*32;
half = csi_matrix_size*bw_size;
all2 = 2*csi_matrix_size*bw_size;

HD_1 = Hur_down(:,1:half);% real part
HD_2 = Hur_down(:,(half+1):all2);% imaginary part
HU_1 = Hur_up(:,1:half);% real part
HU_2 = Hur_up(:,(half+1):all2);% imaginary part
HD_mag = HD_1.^2 + HD_2.^2;
HD_mag1 = sqrt(HD_mag); % downlink magnitude
HU_mag = HU_1.^2 + HU_2.^2;
HU_mag1 = sqrt(HU_mag); % uplink magnitude

HD_phase = angle(HD_1 + 1j*HD_2);% downlink phase
phase_val = HD_phase(70001:100000,:);% downlink phase for test set

%% read the decoded data from CsiNet,DualNet-ABS and DualNet-MAG, and denormalize these data
%similar for U2D-ORG,U2D-ABS,U2D-MAG,

%CsiNet
down1 = csvread('result_indoor/decoded_csinet_indoor5351_bw20.csv');
down1 = down1.*(max(Hur_down(:)) - min(Hur_down(:)));
down1 = down1 + min(Hur_down(:));

%DualNet-ABS
%We use the generalized feature scaling normalization to bring all values into the range [0.5,1]
abs1 = csvread('result_indoor/decoded_dualnet_abs_indoor5351_bw20.csv');
abs1(find(abs1<0.5))=0.5; 
abs1 = abs1.*2 - 1;
abs1 = abs1.*(max(abs(Hur_down(:)))-min(abs(Hur_down(:))));

%DualNet-MAG
%We use the generalized feature scaling normalization to bring all values into the range [0.5,1]
mag = csvread('result_indoor/decoded_dualnet_mag_indoor5351_bw20.csv');
mag2 = mag;%for magnitude dependent phase quantization
mag(find(mag<0.5))=0.5; 
mag = mag.*2 - 1;
mag = mag.*(max(HD_mag1(:)) - min(HD_mag1(:)));
mag = mag + min(HD_mag1(:));

%% Magnitude dependent phase quantization and dequantization
unquan_flag = 1;% 1: no quantization 0: magnitude dependent phase quantization (MDPQ)
step_length = 2*pi/16;%4bit
if unquan_flag == 1
    Step_qua = 1000*ones(30000,half);
else
    Step_qua = 8*ones(30000,half);
    %magnitudes corresponding to CDF value of 0.5, 0.7, 0.8, and 0.9, respectively. 
    %UEs are randomly positioned in the square area in simulation. Base station is positioned at the center of the square area. 
    %This way causes more CSI in small magnitude.
    ancher4 = 0.50725;
    ancher3 = 0.5036;
    ancher2 = 0.50225;
    ancher1 = 0.5012;
      
    Step_qua(mag2<ancher4)= 4;
    Step_qua(mag2<ancher3)= 2;
    Step_qua(mag2<ancher2)= 1;
    Step_qua(mag2<ancher1)= 1/2;
    
    total_num = length(mag2(:));
    len0 = sum(mag2(:)>=ancher4);
    len1 = sum(mag2(:)<ancher4);
    len2 = sum(mag2(:)<ancher3);
    len1 = len1 - len2;
    len3 = sum(mag2(:)<ancher2);
    len2 = len2 - len3;
    len4 = sum(mag2(:)<ancher1);
    len3 = len3 - len4;
    len_mean = (len0*7 + len1*6 + len2*5 + len3*4 + len4*3)./total_num;%mean length for quantized phase
end


phase_qtz = round(phase_val./(step_length./Step_qua));%magnitude dependent phase quantization
phase_val_deqtz = step_length./Step_qua.*phase_qtz;%Dequantization

mag_rec = zeros(30000,all2);
mag_rec(:,1:half) = mag.*cos(phase_val_deqtz);
mag_rec(:,(half+1):all2) = mag.*sin(phase_val_deqtz);

%% MSE and NMSE calculation
power = sum(abs(HD_test).^2,2);
mse_down = sum(abs(HD_test - down1).^2,2);
mse_abs = sum(abs(abs(HD_test) - abs1).^2,2);
mse_mag = sum(abs(HD_test - mag_rec).^2,2);

disp('NMSE for CsiNet:')
10*log10(mean(mse_down./power))
disp('NMSE for DualNet-ABS:')
10*log10(mean(mse_abs./power))
disp('NMSE for DualNet-MAG:')
10*log10(mean(mse_mag./power))

