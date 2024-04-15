% Data Upload 
dati = readtable('SP500_2022.xlsx');

df = dati(:, {'time', 'AAPL_OQ'});

% Convert Time to a Datetime Format 
df.time = datetime(df.time)

% Return Trend 
rt = price2ret(df.AAPL_OQ)
figure()
plot(df.time(2:end), rt)
ylabel('Returns')
xlabel('Dates')
legend('Prices')
title('Return Trend')

% Price Trend
figure()
plot(df.time, df.AAPL_OQ)
ylabel('Prices')
legend('Price Trend')
title('Price Trend')


%% Garch Models
Md1 = garch(1,1);
Md1.Distribution = 't';
[EstMd1_garch,EstParamCov,logL,info]=estimate(Md1,rt);
summarize(EstMd1_garch)

Md2 = garch(1,1);
Md2.Distribution = 'Gaussian';
[EstMd2_garch,EstParamCov,logL,info]=estimate(Md2,rt);
summarize(EstMd2_garch)
 

% Leverage Control
V_cond = infer(EstMd1_garch,rt);
Devst_cond = sqrt(V_cond);


figure('Name','Garch(1,1')
subplot(3,1,1);
plot(df.time(2:end),rt)
title('Returns','color','red')
subplot(3,1,2),
plot(df.time(2:end),rt.^2)
title('Squared Returns','color','red')
subplot(3,1,3);
plot(df.time(2:end), Devst_cond)
title('Conditional Standard Deviation','color','red')


% Standardized Residuals
restd = rt./Devst_cond;
plot(df.time(2:end), restd) 

autocorr(restd) %non c'è evidenza di autocorrelazioni forti
autocorr(restd.^2) %il volatility clustering è stato colto abbastanza bene.
crosscorr(restd, restd.^2)

%% GJR-GARCH Models
Mdl = gjr(1,1);
Mdl.Distribution = 't';
[EstMdl_gjr, EstParamCov,logL,info] = estimate(Mdl,rt);
summarize(EstMdl_gjr)

% Standardized Residuals
Vcond_gjr = infer(EstMdl_gjr,rt);
Devstd_cond_gjr = sqrt(Vcond_gjr);
restd_gjr = rt./Vcond_gjr;
figure('Name','Standardizer Residuals')
plot(df.time(2:end),restd_gjr)

% Autocorrelation Function
autocorr(restd_gjr) 
autocorr(restd_gjr.^2)
% Cross-Correlation Function
crosscorr(restd_gjr,restd_gjr.^2)

% Volatility Forecast
L = 5;
l = length(rt);
for1 = forecast(EstMd1_garch,L,'Y0',rt);

figure()
plot(V_cond,'color',[.7,.7,.7])
xlim([5500 l+L+5]);
hold on
plot(l+1:l+L,for1,'r','LineWidth',2);
legend('Conditional Variance','Forecast')
title('Volatility Forecast with T-Garch')
hold off

figure()
plot(Vcond_gjr,'color',[.7,.7,.7])
xlim([5500 l+L+5]);
hold on
plot(l+1:l+L,for1,'r','LineWidth',2);
legend('Conditional Variance','Forecast')
title('Volatility Forecast with GJR-Garch')
hold off

