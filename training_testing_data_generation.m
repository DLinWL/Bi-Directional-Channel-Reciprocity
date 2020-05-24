load('mat_indoor5351_bw20MHz_down.mat')
load('mat_indoor5351_bw20MHz_up.mat')

org_up = Hur_up;
org_down = Hur_down;
abs_up = abs(Hur_up);
abs_down = abs(Hur_down);


H_up_n1 = norm_H3(org_up);
H_down_n1 = norm_H3(org_down);

HD_train = H_down_n1(1:70000,:);
save Data100_Htrainin_down_FDD.mat HD_train;
HD_val = H_down_n1(70001:100000,:);
save Data100_Hvalin_down_FDD.mat HD_val;

HU_train = H_up_n1(1:70000,:);
save Data100_Htrainin_up_FDD.mat HU_train;
HU_val = H_up_n1(70001:100000,:);
save Data100_Hvalin_up_FDD.mat HU_val;

H_up_n2 = norm_H2(org_up);
H_down_n2 = norm_H2(org_down);

HD_train = H_down_n2(1:70000,:);
save Data100_Htrainin_down_FDD2.mat HD_train;
HD_val = H_down_n2(70001:100000,:);
save Data100_Hvalin_down_FDD2.mat HD_val;



HU_train = H_up_n2(1:70000,:);
save Data100_Htrainin_up_FDD2.mat HU_train;
HU_val = H_up_n2(70001:100000,:);
save Data100_Hvalin_up_FDD2.mat HU_val;

HD_1 = org_down(:,1:1024);
HD_2 = org_down(:,1025:2048);
HU_1 = org_up(:,1:1024);
HU_2 = org_up(:,1025:2048);
HD_mag_tmp = HD_1.^2 + HD_2.^2;
HD_mag = sqrt(HD_mag_tmp);
HU_mag_tmp = HU_1.^2 + HU_2.^2;
HU_mag = sqrt(HU_mag_tmp);

H_down_n3 = norm_H2(HD_mag);
H_up_n3 = norm_H2(HU_mag);


HD_train = H_down_n3(1:70000,:);
HU_train = H_up_n3(1:70000,:);
save Data100_Htrainin_mag.mat HD_train HU_train;
HD_val = H_down_n3(70001:100000,:);
HU_val = H_up_n3(70001:100000,:);
save Data100_Hvalin_mag.mat HD_val HU_val;

