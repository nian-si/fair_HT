T = 2000;
rec=zeros(T,1);
empirical_var = zeros(T,1);
d = 2;
tau = 0.5;
theta = [0,1]';
Nlist = [30,100,500,1000,2000];
p11 = 0.4;
p01 = 0.1;
p10 = 0.4;
p00 = 0.1;


alphalist = [0.1,0.05,0.01];
for N = Nlist
    N
    var = compute_variance(p11 , p01 , p10, p00);
    
    parfor t = 1:T
        X2 = normrnd(0,sqrt(5),N,1);
        rnd_A1 = normrnd(6,sqrt(3.5),N,1);
        rnd_A0Y1 = normrnd(-2,sqrt(5),N,1);
        rnd_A0Y0 = normrnd(-4,sqrt(5),N,1);
        p_val = rand(N,1);
        A = (p_val<p11) + (p_val<p11 + p01 + p10 & p_val>=p11 + p01 );
        Y = (p_val<p11 + p01);
        X1 = A.*rnd_A1 +  Y.*rnd_A0Y1.*(1-A) + (1-Y).*rnd_A0Y0.*(1-A);
        X = [X1,X2];
        phi_func = @(u1,u2)(u1/mean(u1) - u2/mean(u2));
        phi_derivative = @(u1,u2)([-u1/mean(u1)^2,  u2/mean(u2)^2]);
        rec(t) = discontinuous_RWPI(X,A,Y,theta,N,tau,phi_func );
        empirical_var(t) = compute_empirical_variance(X,A,Y,theta,tau,phi_func,phi_derivative);
        
       
    end
    
   
    
    
    [mean(rec),var,mean(empirical_var)]

   
  
    figure
    hold on
    
    threshold = empirical_var * chi2inv(1-alphalist ,1);
    [sum(rec>threshold(:,1)) / T,sum(rec>threshold(:,2)) / T,sum(rec>threshold(:,3)) / T]
    plot_max = 3;
    x = 0:0.01:plot_max;
    y = chi2pdf(x/var,1)/var;  
    
    
    plot(x,y,'linewidth',4);
    
    histogram(rec(1:T),0:0.1:plot_max,'Normalization','pdf')
    xlim([0,plot_max]);
    ylabel('density');
   % xlabel('$N \mathcal{D}(\bf{\hat{P}}^N)$','interpreter','latex');
    set(gca,'fontsize',30,'fontname','Times');
set(gcf, 'position', [0 0 678 568]);
set(gcf, 'PaperPositionMode', 'auto');
    outfile=['N=',num2str(N),'.eps'];
    print(gcf,'-depsc',outfile);
    %saveas(gcf,[name,'.fig']);
    %plot(x,chi2pdf(x,df),'linewidth',1.5);
    hold off
    
    %plot(x,chi2pdf(x,df),'linewidth',1.5);
%   
end