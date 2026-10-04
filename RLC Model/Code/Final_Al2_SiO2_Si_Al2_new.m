clear all;
% close all;
clc;
inputdata = xlsread('Al2-SiO2-Al2.xlsx');
ii = 1;
fstart = 1;
fend = 3601;
f = inputdata(fstart:fend ,1,1);
wl = 299.79./f * 1e-6;
data1 = inputdata(fstart:fend ,:);;
dhp = [1.0907	0.2	0.1	    1.776
1.0907	0.2	        0.05	1.776
1.0907	0.2	        0.01	1.776
1.0907	0.2233	    0.1	    2
1.0907	0.2233	    0.05	2
1.0907	0.2233	    0.01	2
1.1086	0.25    	0.1	    2.3811
1.1086	0.25	    0.05	2.3811
1.1086	0.25	    0.01	2.3811
1.0907	0.3	        0.1 	2.806
1.0907	0.3	        0.05	2.806
1.0907	0.3	        0.01	2.806
1.0816	0.3072	    0.1 	3
1.0816	0.3072	    0.05	3
1.0816	0.3072	    0.01	3
1.3637	0.2318	    0.1	    2
1.3637	0.2318	    0.05	2
1.3637	0.2318	    0.01	2
1.394	0.25	    0.1	    2.2681
1.394	0.25	    0.05	2.2681
1.394	0.25	    0.01	2.2681
1.2445	0.3	        0.1	    2.7626
1.2445	0.3	        0.05	2.7626
1.2445	0.3	        0.01	2.7626
1.4085	0.3152	    0.1  	3
1.4085	0.3152	    0.05	3
1.4085	0.3152	    0.01	3
1.6043	0.25	    0.1 	1.9911
1.6043	0.25	    0.05	1.9911
1.6043	0.25	    0.01	1.9911
1.6008	0.2503   	0.1	    2
1.6008	0.2503	    0.05	2
1.6008	0.2503	    0.01	2
1.5543	0.3	        0.1 	2.6986
1.5543	0.3	        0.05	2.6986
1.5543	0.3	        0.01	2.6986
1.4968	0.3073	    0.1 	3
1.4968	0.3073	    0.05	3
1.4968	0.3073	    0.01	3
1.7186	0.3213	    0.1 	3
1.7186	0.3213	    0.05	3
1.7186	0.3213	    0.01	3           %%
1.9478	0.25	    0.1 	2.4039
1.9478	0.25	    0.05	2.4039
1.9478	0.25	    0.01	2.4039
1.9512	0.3	        0.1	    2.5991
1.9512	0.3	        0.05	2.5991
1.9512	0.3	        0.01	2.5991
1.85	0.3	        0.1 	2.6822
1.85	0.3	        0.05	2.6822
1.85	0.3	        0.01	2.6822
1.991	0.3294   	0.1	    3
1.991	0.3294	    0.05	3
1.991	0.3294	    0.01	3
]*1e-6;


% p = 3.1E-6;
% d = 1300E-9;
% h0 = 100E-9;
% h = 400E-9;
%  dhp = [1.3 0.4 0.1 3.1]*1e-6;
% dhp = [1.08 0.21 0.01 3]*1e-6;
 ii = 48;
nd = dhp(ii,1);  %1 ~ 5;     %h=3 p =4 is best   h3 p3
nh = dhp(ii,2);  %1 ~ 5;     %h=3 p =4 is best   h3 p3 
nh2 = dhp(ii,3);  %1 ~ 5;     %h=3 p =4 is best   h3 p3 p10;  %1 ~ 25;
np = dhp(ii,4);  %1 ~ 5;     %h=3 p =4 is best   h3 p3 p10;  %1 ~ 25;


xx = [1.672111576 0.588613726 0.2];
%xx = [1.166 4.505497 0.519 0.093732];  % 2.196/22.4

%% find the peak area between 0.5*max ~ max
% data2 = data1;
% Gr1 = diff(data2(:,ii+1));
% im = 1;
% while Gr1(im)>0
%   im = im +  1;
% end
% keyfactor = 0.75;
% [max0 max1] = max(diff(data2(1:im,ii+1)));
% ij = 1;
% while Gr1(ij)< (max0 *keyfactor)
%     ij = ij + 1;
% end
% [min0 min1] = min(diff(data2(max1:im*2-ij,ii+1)));
% ik = max1+min1;
% while Gr1(ik)< (min0 *keyfactor)
%     ik = ik + 1;
% end
% data = 0;
% fstart = ij;
% fend = ik;
% f = inputdata(fstart:fend ,1,1);
% wl = 299.79./f * 1e-6;

data1 = inputdata(fstart:fend ,:);

Al = xlsread('Al2.xlsx');
Alf = Al(:,1)/1e12;
Alr = Al(:,2);
Ali = Al(:,3);

[fitresult, gof] = createFit(Alf,Alr);
Alreal = fitresult(f);

[fitresult, gof] = createFit(Alf,Ali);
Alimage = fitresult(f);

Ale = Alreal - i*Alimage;

SiO2 = xlsread('SiO2.xlsx');
SiO2f = SiO2(:,1)/1e12;
SiO2r = SiO2(:,2);
SiO2i = SiO2(:,3);

[fitresult, gof] = createFit(SiO2f,SiO2r);
SiO2real = fitresult(f);

[fitresult, gof] = createFit(SiO2f,SiO2i);
SiO2image = fitresult(f);

SiO2e = SiO2real - i*SiO2image;

Si = xlsread('Sipermittivity.xlsx');
Sif = Si(:,1)/1e12;
Sir = Si(:,2);
Sii = Si(:,3);

[fitresult, gof] = createFit(Sif,Sir);
Sireal = fitresult(f);

[fitresult, gof] = createFit(Sif,Sii);
Siimage = fitresult(f);

%esi = 11.9;   % real part of silicon
%esi = 16;   % real part of silicon
esi = Sireal - i*Siimage; 

Z0 = 377;
c = 2.9979e8;                                            %speed of light
w = c./wl*2*pi;                                     %angle frequency

% newAl2   1.3926084526627913	3.551131390390778

options = optimoptions('lsqcurvefit','Display','none'); 
Q1 = [1,    4];
Q2 = [0.1,  1];
Q3 = [10,  10];       
Alf = Alf*1e12;
[x3,error1] = lsqcurvefit(@(x,t0) (x(1)*1e16)^2*(x(2)*1e-15)./(2*pi*Alf.*(1+(2*pi*Alf*(x(2)*1e-15)).^2)),Q1,Alf,Ali,Q2,Q3,options);

wp = x3(1) * 1e16;
tmetal= x3(2) * 1e-15;

% wp = 1.3926084526627913e16;  
% tmetal = 3.551131390390778e-15; % relaxation time taoau

eer = 1 - wp^2./(w.*(w + i/tmetal));
e0 = 8.854e-12; % permittivity of free space
u0 = 4*pi*1e-7;  % permability of free space
tau = 75e-9;    % the thickness of metal disk
k = (((Alreal.^2 + Alimage.^2).^0.5 - Alreal)/2).^0.5;
deltaau = wl./(2*pi*k); % pentration depth
sigmaal = e0*wp*wp*tmetal;  % DC conductivity

save('optm2.mat');

    for h1 = nh                  % check it is 1 ~ 5 or 1 ~ 4
        for p1 = np
            for d1 = nd 
                for h2 = nh2
                    
                p = p1;
                hsi = h1 - h2;
                hsio = h2;
                 p = p1;
                hsi = h1 - h2;
                hsio = h2;
                c2 = xx(1)*exp((d1 - p*xx(2))*1e6);
                c4 = xx(3);
                c1 = 1/c2;
                c3 = 1/c2;                      

                Cm = c1.*e0.*pi*(d1.*0.5).^2./(4.*(hsi./esi + hsio./SiO2e));
                Lm = u0.*h1.*c2/2;                                                     %single Lm
                 Ce = 3.*c3.* e0.*pi.*(d1.*0.5).*h1./((p - d1)); 
%                 Ce = 1.*c3.* e0.*pi.*(d1.*0.5).*h1./(4*(p - d1)); 
%                Ce = c3.*e0.*esi.*pi.*(d1.*0.5).^2./(esi.^0.32.*p.*4); 
                
                Rc = c2./(deltaau.*sigmaal);
                Lc = c2./(deltaau.*e0.*wp.^2);
                Rg = c2.*c4./(deltaau.*sigmaal);
                Lcg = c2.*c4./(deltaau.*e0.*wp.^2);
                
                
              %  Ce = 8*e0.*esi.*(pi.*(d(d1).*0.5).^2./4)./(c1.*pi^3/32*esi.^0.25.*(deltaau./(c2.*1e-5)).^0.5.*p);
                               
                Lt = Lc + Lm;
                Lg = Lcg + Lm;              
                Zt = Lt.*w.*j + Rc;
                Zb = 2./(Cm.*j.*w) + Lg.*w.*j + Rg;  %  + Zin;
                Ze = 1./(Ce.*j.*w);
                
                
                Zcross = 1./(1./Zb + 1./Zt + 1./Ze);% + Zin;
                
%                 Zs = 1j.* (u0./(e0 .* esi)).^0.5 .* tan(w.* (e0*u0.*esi).^0.5 * h1); 
%                 Zs1 = (u0./(e0 .* esi)).^0.5;
%                 Zal = (u0./(e0 .* Ale)).^0.5;
%                 Z1 = Zs1 .*(Zal + Zs)./(Zs1 + 1j.* Zal .* tan((w.* (e0*u0.*esi).^0.5 * h1))); 
%                   Zcross = 1./(1./Zb + 1./Zt + 1./Ze) + Z1;

                As123 = 1-(abs((Zcross-Z0)./(Zcross+Z0))).^2; 
                ssres = sum((As123 - data1(:,ii+1)).^2);
                sstot = sum((As123 - mean(As123)).^2);
                Rsquare = 1 - ssres./sstot;    % h p d
                end
            end
        end
    end

 
 plot(f,As123,f,data1(:,ii+1),'o')
 tt1 = data1(:,ii+1);
 wll = wl.*1e6;   
plot(wll,As123,wll,data1(:,ii+1),'o');axis([5 15 0 1])
 
 