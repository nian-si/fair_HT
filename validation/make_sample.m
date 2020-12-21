function [N] = make_sample (num1,df1,num2,df2,edges)
n=1000000;
data =num1* chi2rnd(df1*ones(n,1),n,1)+num2*chi2rnd(df2*ones(n,1),n,1);
pdSix = fitdist(data,'Kernel','BandWidth',2);
N=pdf(pdSix,edges);
end