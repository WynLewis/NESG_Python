function [Output] = SchulmanNominalRateSimulator64(DailyHistoricalScore, DailyForecastScore, NumSims, dt, Tmax, OutputInterval)

        %DailyHistoricalScore contains three columns for SCORE1, SCORE2, and
        %SCORE 3.  Rows are in chronological order from oldest to T0.
        
        %DailyForecastScore contains four columns.  The first column is the
        %time index (in "years" units) for the forecast dates, beginning at
        %zero.  The remaining columns are the SCOREs for the forecast yield
        %curves.
               
        %The last row of DailyHistoricalScore and first row of
        %DailyForecastScore are identical and correspond to initial
        %conditions.
        
        %To run with no history and/or with no forecast, enter just one row
        %in DailyHistoricalScore and/or DailyForecastScore corresponding to
        %initial conditions.

        
        %INPUT:  Arbitrary points on yield curve to model.
        T=[1/52 (1:6)/12 .75 1:20 25 30];  
        

        
        % Nominal Rates
        CurveParams =   [0.25173673	0.41615380	0.04809383	0.02427488	0.31618294	0.40000000	0.55000000
						0.74827338	0.16527153	-0.01643883	0.01172933	-0.00418756	-0.90000000	0.20000000
						0.04029944	-0.05889147	-0.00002636	0.00002622	0.00717977	-0.02500000	-1.50000000
						0.05674044	0.25442796	0.15708250	0.37414145	0.28443789	0.25000000	0.90000000
						0.89249482	0.71550848	0.94578667	0.59996753	1.00000000	1.00000000	0.55000000]';

        sigma =    		[ 3.0947925	0.5	0.15 ];   %Steady State Volatility Parameters
        eta =      		[ 1.1422149	0.5	0.15 ];  %Process Volatility Parameters     

        %INPUT:  Lognormal Parameters specifying long-run distribution of rates at each point on the curve.            
        rate_omega  = repmat(     CurveParams(1,1) + (CurveParams(1,2) + CurveParams(1,3)*T).*exp(-CurveParams(1,4)*T.^CurveParams(1,5))  ,NumSims,1);
        rate_beta   = repmat(     CurveParams(2,1) + (CurveParams(2,2) + CurveParams(2,3)*T).*exp(-CurveParams(2,4)*T.^CurveParams(2,5))  ,NumSims,1);
        rate_mu     = repmat(     CurveParams(3,1) + (CurveParams(3,2) + CurveParams(3,3)*T).*exp(-CurveParams(3,4)*T.^CurveParams(3,5))  ,NumSims,1);
        rate_sigma  = repmat(     CurveParams(4,1) + (CurveParams(4,2) + CurveParams(4,3)*T).*exp(-CurveParams(4,4)*T.^CurveParams(4,5))  ,NumSims,1);
        
        %INPUT:  Factors for shift-tilt-flex model at each point on the curve.     . 
        F(1,:)= CurveParams(5,1) + (CurveParams(5,2) + CurveParams(5,3)*T).*exp(-CurveParams(5,4)*T.^CurveParams(5,5)) ;
        F(2,:)= CurveParams(6,1) + (CurveParams(6,2) + CurveParams(6,3)*T).*exp(-CurveParams(6,4)*T.^CurveParams(6,5)) ;
        F(3,:)= CurveParams(7,1) + (CurveParams(7,2) + CurveParams(7,3)*T).*exp(-CurveParams(7,4)*T.^CurveParams(7,5)) ;          

        clear CurveParams

        %INPUT:  CMF Pareto Parameters
        ParetoParamvec = [20.0 80  0.7750  0.3];       
     
        RateAlpha       = ParetoParamvec(1);
        RateTheta       = ParetoParamvec(2);
        IncrementSigma  = ParetoParamvec(3);
        IncrementGamma  = ParetoParamvec(4);
        
        %Pre-compute expected number of increments in place at a given time
        N = ceil(RateTheta/(RateAlpha-1)/dt);
        
        %Pre-compute schulman distribution normalization constants
        SchulmanMean   = (2* rate_omega-1).*exp(0.5*rate_beta.^2);
        SchulmanVol    = sqrt( (2* rate_omega.^2 - 2*rate_omega+1).*exp(2*rate_beta.^2) -  (2*rate_omega-1).^2.*exp(rate_beta.^2) -  2*rate_omega.*(1- rate_omega) );
        
        %%%%%%%%%%%%%%%%%%%
        %SET INITIAL RANDOM STACK
        %This will age-off if you provide a history, or will be be the
        %inital random stack if you provide no history.
        %%%%%%%%%%%%%%%%%%%
        
        %compute simulation constants
        kappa2    = -.5*(eta(2)/sigma(2))^2;
        kappa3    = -.5*(eta(3)/sigma(3))^2;

        %Initialize state variables for t0
        Output      = zeros(NumSims*(Tmax/dt/OutputInterval + 1),35);

        SCORE       =  repmat(DailyHistoricalScore(end,:)',1,NumSims);
        
        L=logninv(normcdf(SCORE'*F,0,1),0,rate_beta);
        SchulmanVariate     =(rate_omega.*L  -  (1- rate_omega)./L  -  SchulmanMean)    ./ SchulmanVol;
                            
        rate        = rate_mu + ...
                      rate_sigma .* SchulmanVariate;

        %Initialize state variables for t0
        NumHistoricalEpochs      = size(DailyHistoricalScore,1);
%         NumForecastEpochs        = size(DailyForecastScore,1);
        NumEpochs                = NumHistoricalEpochs+Tmax/dt;

        StackSize               = N;                              

        RateState               = zeros(NumEpochs,NumSims);

        %Then apply the increments to develop the random initial conditions
        %Apply a correction factor to sum to proper initial score.
        for j=1:NumSims
            
            %for each increment in stack, determine expiration date and vol
            IncrementVol              = zeros(1,StackSize);
            RateExpiryPointer         = zeros(1,StackSize);
            
            for i=1:StackSize
                    InForceDuration      = RateTheta*( rand^-(1/(RateAlpha-1))  -  1 );
                    ExcessDuration      = (RateTheta + InForceDuration)*( rand^-(1/RateAlpha)  -  1 );
                    IncrementVol(i)      = IncrementSigma*(InForceDuration+ExcessDuration).^IncrementGamma;
                    RateExpiryPointer(i) = min(NumEpochs,ceil((ExcessDuration)/dt));
            end
            
            Increments = IncrementVol*randn(StackSize)/StackSize;
            Increments = Increments + (DailyHistoricalScore(1,1) - sum(Increments))/StackSize;
            for i=1:StackSize
                RateState(1:RateExpiryPointer(i),j) = RateState(1:RateExpiryPointer(i),j) + Increments(i);
            end
        end

        %%%%%%%%%%%%%%%%%%%
        %PROCESS HISTORICAL EXPERIENCE
        %The indexing and data structure means that you always at least process the initial
        %condition, but the dW calculation will set the increment to zero
        %so no action is taken on the initial condition record.
        %%%%%%%%%%%%%%%%%%%  
        for Epoch=1:NumHistoricalEpochs

            RateExpiry              = RateTheta*((1-rand(1,NumSims)).^(-1/(RateAlpha-1))-1);

            RateExpiryPointer       = min(NumEpochs,Epoch+ceil(RateExpiry/dt));
            for j=1:NumSims
                dW = (DailyHistoricalScore(Epoch,1)-RateState(Epoch,j));
                RateState(Epoch:RateExpiryPointer(j),j) = RateState(Epoch:RateExpiryPointer(j),j) + dW;
            end
        end        
        
%         ExpectedState = mean(RateState');
                
        %%%%%%%%%%%%%%%%%%%
        %SIMULATION LOOP
        %%%%%%%%%%%%%%%%%%%
        Epoch                   = NumHistoricalEpochs;
        write_pointer           = 1+(Epoch-NumHistoricalEpochs)/OutputInterval : 1+(NumEpochs-NumHistoricalEpochs)/OutputInterval: NumSims*((NumEpochs-NumHistoricalEpochs)/OutputInterval+1);
        Output(write_pointer,:) = [(1:NumSims)' (Epoch-NumHistoricalEpochs)*dt*ones(NumSims,1) SCORE' rate];
        OutputCounter           = 1;
            

        ForecastAdjustmentScore1 = 0;    
        for Epoch=NumHistoricalEpochs+1:NumEpochs
            
            RateExpiry              = RateTheta*((1-rand(1,NumSims)).^(-1/(RateAlpha-1))-1);
            IncrementVol            = IncrementSigma*(RateExpiry).^IncrementGamma;


            RateExpiryPointer       = min(NumEpochs,Epoch+ceil(RateExpiry/dt));
            ForecastTime       = (Epoch-NumHistoricalEpochs)*dt;
            if ForecastTime>=DailyForecastScore(end,1)
                ForecastAdjustmentScore1 = ForecastAdjustmentScore1.*exp(-.0006);
            end
            
            for j=1:NumSims
                RateState(Epoch:RateExpiryPointer(j),j) = RateState(Epoch:RateExpiryPointer(j),j) + IncrementVol(j)*randn*sqrt(dt);

                %%%%%%%%%%%%%%%%%%%
                %FORECAST SUB-LOOP
                %If the projection period fall within the forecast period,
                %then subtract out the expected level of SCORE1 (assuming
                %no forecast) and add the forecast level back in.
                %%%%%%%%%%%%%%%%%%%  
                
                
                
                if ForecastTime<=DailyForecastScore(end,1)
                    ForecastAdjustmentScore1 = (interp1(DailyForecastScore(:,1),DailyForecastScore(:,2),ForecastTime)) - ...
                                          (DailyForecastScore(1,2)*((RateTheta+ForecastTime)/RateTheta)^(1-RateAlpha));
					RateState(Epoch,j) = RateState(Epoch,j) + ForecastAdjustmentScore1;
                
                else
                    RateState(Epoch,j) = RateState(Epoch,j) + ForecastAdjustmentScore1;                    
                end
   
            end
            
            %%%%%%%%%%%%%%%%%%%
            %FORECAST SUB-LOOP
            %If the projection period fall within the forecast period,
            %then subtract out the expected change in SCORE2 and SCORE3 (assuming
            %no forecast) and add the forecast change back in.  This is
            %exactly accurate because the O-U processes are memoryless.
            %%%%%%%%%%%%%%%%%%%  

            if ForecastTime<=DailyForecastScore(end,1)

                ForecastAdjustment = (interp1(DailyForecastScore(:,1),DailyForecastScore(:,3),ForecastTime) - interp1(DailyForecastScore(:,1),DailyForecastScore(:,3),ForecastTime-dt)) - ...
                                     (interp1(DailyForecastScore(:,1),DailyForecastScore(:,3),ForecastTime-dt)*exp(kappa2*dt)-interp1(DailyForecastScore(:,1),DailyForecastScore(:,3),ForecastTime-dt));
                SCORE(2,:) = SCORE(2,:) + ForecastAdjustment;
                
                ForecastAdjustment = (interp1(DailyForecastScore(:,1),DailyForecastScore(:,4),ForecastTime) - interp1(DailyForecastScore(:,1),DailyForecastScore(:,4),ForecastTime-dt)) - ...
                                     (interp1(DailyForecastScore(:,1),DailyForecastScore(:,4),ForecastTime-dt)*exp(kappa3*dt)-interp1(DailyForecastScore(:,1),DailyForecastScore(:,4),ForecastTime-dt));
                SCORE(3,:) = SCORE(3,:) + ForecastAdjustment;
            end
          
            SCORE(1,:)  = RateState(Epoch,:);
            SCORE(2,:)  = SCORE(2,:) + kappa2 * SCORE(2,:) *dt  + eta(2)*randn(1,NumSims)*sqrt(dt);
            SCORE(3,:)  = SCORE(3,:) + kappa3 * SCORE(3,:) *dt  + eta(3)*randn(1,NumSims)*sqrt(dt);   

            L=logninv(normcdf(SCORE'*F,0,1),0,rate_beta);
            SchulmanVariate     =(rate_omega.*L  -  (1- rate_omega)./L  -  SchulmanMean)    ./ SchulmanVol;

            rate        = rate_mu + ...
                          rate_sigma .* SchulmanVariate;

            %Generate pointer values and write to output variable
            if OutputCounter==OutputInterval
                write_pointer           = 1+(Epoch-NumHistoricalEpochs)/OutputInterval : 1+(NumEpochs-NumHistoricalEpochs)/OutputInterval: NumSims*((NumEpochs-NumHistoricalEpochs)/OutputInterval+1);
                Output(write_pointer,:) = [(1:NumSims)' (Epoch-NumHistoricalEpochs)*dt*ones(NumSims,1) SCORE' rate];
                OutputCounter           = 1;       
            else
                OutputCounter=OutputCounter+1; 
            end
        end
