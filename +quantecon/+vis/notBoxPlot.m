function varargout=notBoxPlot(y,x,varargin)
% notBoxPlot - Doesn't plot box plots!
%
% function notBoxPlot(y,x,'Param1',val1,'Param2',val2,...)
%
%
% Purpose
% An alternative to a box plot, where the focus is on showing raw
% data. Plots columns of y as different groups located at points
% along the x axis defined by the optional vector x. Points are
% layed over a 1.96 SEM (95% confidence interval) in red and a 1 SD
% in blue. The user has the option of plotting the SEM and SD as a
% line rather than area. Raw data are jittered along x for clarity. This
% function is suited to displaying data which are normally distributed.
% Since, for instance, the SEM is meaningless if the data are bimodally
% distributed.
%
% Ported to quantecon.vis. Original by Rob Campbell.

% Check input arguments
if nargin==0
    help(mfilename)
    return
end

% Check if Y is of a suitable class
if ~isnumeric(y) && ~istable(y) && ~isa(y,'LinearModel')
    fprintf('Variable y is a %s. This is not an allowed input type. see help %s\n',...
        class(y), mfilename)
    return
end

% Parse the different call types
modelCIs=[];
tableOrModelCall=false;

switch lower(class(y))

    case 'table'
        tableOrModelCall=true;
        if nargin>1 %so user doesn't need to specify a blank variable for x
            if ~isempty(x)
                varargin=[x,varargin];
            end
        end
        thisTable=y;
        varNames=thisTable.Properties.VariableNames;
        if length(varNames) ~= 2
            fprintf('% s can only handle tables with two variables\n',mfilename)
            return
        end
        y = thisTable.(varNames{1});
        x = thisTable.(varNames{2});

    case 'linearmodel'
        tableOrModelCall=true;
        if nargin>1 %so user doesn't need to specify a blank variable for x
            if ~isempty(x)
                varargin=[x,varargin];
            end
        end

        thisModel=y;

        if length(thisModel.PredictorNames) >1
            fprintf('% s can only handle linear models with one predictor\n',mfilename)
            return
        end
        y = thisModel.Variables.(thisModel.ResponseName);
        x = thisModel.Variables.(thisModel.PredictorNames{1});

        %Check that x is of a suitable type
        if isnumeric(x)
            fprintf('The model predictor variable should not be continuous\n')
            return
        end
        if iscell(x)
            fprintf('Coercing predictor variable from a cell array to a categorical variable\n')
            x=categorical(x);
        end

        varNames = {thisModel.ResponseName,thisModel.PredictorNames{1}}; %for the axis labels

        % Set the SD bar to have 1.96 standard deviations
        varargin = [varargin,'numSDs',1.96];

        % Get the the confidence intervals from the model
        modelCIs = coefCI(thisModel,0.05);

    otherwise %Otherwise Y is a vector or a matrix

        if isvector(y)
            y=y(:);
        end

        % Handle case where user doesn't supply X, but there are user-supplied param/val pairs. e.g.
        % notBoxPlot(rand(20,5),'jitter',0.5)
        if nargin>2 && ischar(x)
            varargin=[x,varargin];
            x=[];
        end

        % Generate an monotonically increasing X variable if the user didn't supply anything
        % for the grouping variable
        if nargin<2 || isempty(x)
            x=1:size(y,2);
        end

end %switch class(y)


%If x is logical then the function fails. So let's make sure it's a double
x=double(x);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse input arguments
params = inputParser;
params.CaseSensitive = false;

%User-visible options
params.addParameter('jitter', 0.3, @(x) isnumeric(x) && isscalar(x));
params.addParameter('style','patch', @(x) ischar(x) && any(strncmpi(x,{'patch','line','sdline'},4)) );
params.addParameter('interval','SEM', @(x) ischar(x) && any(strncmpi(x,{'sem','tinterval'},4)) );
params.addParameter('markMedian', false, @(x) islogical(x));

%Options hidden from the user
params.addParameter('numSDs',1, @(x) isnumeric(x) && isscalar(x) && x>=0)
params.addParameter('manualCI',[], @(x) (isnumeric(x) && isscalar(x)) || isempty(x) )

params.parse(varargin{:});

%Extract values from the inputParser
jitter     = params.Results.jitter;
style      = params.Results.style;
interval   = params.Results.interval;
markMedian = params.Results.markMedian;

%The multiplier for the SD patch. e.g. for 1.96 SDs this value should be 1.96
numSDs = params.Results.numSDs;
manualCI = params.Results.manualCI; %Is used by the recursive call to over-ride the CI when y is a LinearModel

%Set interval function
switch lower(interval)
    case 'sem'
        intervalFun = @SEM_calc;
    case 'tinterval'
        intervalFun = @tInterval_calc;
    otherwise
        error('Interval %s is unknown',interval)
end

if jitter==0 && strcmp(style,'patch')
    warning('A zero value for jitter means no patch object visible')
end


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% We now loop through the unique x values, plotting each notBox in turn
% using recursive calls to notBoxPlot.
if isvector(y) && isvector(x) && length(x)>1
    x=x(:);

    if length(x)~=length(y)
        error('length(x) should equal length(y)')
    end

    u=unique(x);
    for ii=1:length(u)
        f = find(x==u(ii));

        %If a model was used, we use the 95% t-intervals it produces
        if ~isempty(modelCIs)
            thisCI = range(modelCIs(ii,:))/2; %the interval is symmetric and we need just this.
        else
            thisCI =[];
        end

        [h(ii),s(ii)]=quantecon.vis.notBoxPlot(y(f),u(ii),varargin{:},'manualCI',thisCI); %recursive call
    end


    %Make plot look pretty
    if length(u)>1
        xlim([min(u)-1,max(u)+1])
        set(gca,'XTick',u)
    end

    if nargout>0
        varargout{1}=h;
    end
    if nargout>1
        varargout{2}=s;
    end

    %If we had a table we can label the axes
    if tableOrModelCall
        ylabel(varNames{1})
        xlabel(varNames{2})
    end

    return % User's call to notBoxPlot never goes beyond here
end
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if length(x) ~= size(y,2)
    error('length of x doesn''t match the number of columns in y')
end

% We're going to render points with the same x value in different
% colors so we loop through all unique x values and do the plotting
% with nested functions. Avoid clearing the axes in order to give
% the user more flexibility in combining plot elements.
hold on
[uX,a,b]=unique(x);

H=[];
stats=[];
for ii=1:length(uX)
    f=b==ii;
    [hTemp,statsTemp]=myPlotter(x(f),y(:,f));
    H = [H,hTemp];
    stats = [stats,statsTemp];
end

hold off

%Tidy up plot: make it look pretty
if length(x)>1
    set(gca,'XTick',unique(x))
    xlim([min(x)-1,max(x)+1])
end


%handle the output arguments
if nargout>0
    varargout{1}=H;
end

if nargout>1
    varargout{2}=stats;
end

%Nested functions follow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [h,statsOut]=myPlotter(X,Y)
        %This is a nested function that shares the caller's namespace

        if isempty(manualCI)
            SEM=intervalFun(Y); %A function handle to a supplied local function
        else
            SEM=manualCI;
        end

        % NaNs do not contribute to the sample size
        if ~any(isnan(Y(:)))
            % So we definitely have no problems with older MATLAB releases or non-stats toolbox installs
            SD=std(Y)*numSDs;
            mu=mean(Y);
            if markMedian
                med = median(Y);
            end
        elseif ~verLessThan('matlab','9.0') %from this version onwards we use the omitnan flag
            SD=std(Y,'omitnan')*numSDs;
            mu=mean(Y,'omitnan');
            if markMedian
                med = median(Y,'omitnan');
            end
        elseif which('nanmean') %Otherwise proceed if stats toolbox is there
            SD=nanstd(Y)*numSDs;
            mu=nanmean(Y);
            if markMedian
                med = nanmedian(Y);
            end
        else %raise error
            error('You have NaNs in your data set but are running older than R2016a or you have no stats toolbox.')
        end

        %The plot colors to use for multiple sets of points on the same x
        %location
        cols=hsv(length(X)+1)*0.5;
        cols(1,:)=0;
        jitScale=jitter*0.55; %To scale the patch by the width of the jitter

        for k=1:length(X)

            thisY=Y(:,k);
            thisY=thisY(~isnan(thisY));
            thisX=repmat(X(k),1,length(thisY));

            %Assemble stats for optional command line output
            statsOut(k).mu = mu(k);
            statsOut(k).interval = SEM(k);
            statsOut(k).sd = SD(k);

            %Add the SD as a patch if the user asked for this
            if strcmp(style,'patch')
                h(k).sdPtch=patchMaker(SD(k),[0.6,0.6,1]);
            end

            %Build patch surfaces for SEM, the means, and optionally the medians
            if strcmp(style,'patch') || strcmp(style,'sdline')
                h(k).semPtch=patchMaker(SEM(k),[1,0.6,0.6]);
                h(k).mu=plot([X(k)-jitScale,X(k)+jitScale],[mu(k),mu(k)],'-r',...
                    'linewidth',2);
                if markMedian
                    statsOut(k).median = med(k);
                    h(k).med=plot([X(k)-jitScale,X(k)+jitScale],[med(k),med(k)],':r',...
                        'linewidth',2);
                end
            end

            % Generate scatter in X
            thisX=violaPoints(thisX,thisY);
            C=cols(k,:);

            h(k).data=plot(thisX, thisY, 'o', 'color', C,...
                'markerfacecolor', C+(1-C)*0.65);
        end  %for k=1:length(X)


        %Plot SD as a line
        if strcmp(style,'line') || strcmp(style,'sdline')
            for k=1:length(X)
                h(k).sd=plot([X(k),X(k)],[mu(k)-SD(k),mu(k)+SD(k)],...
                    '-','color',[0.2,0.2,1],'linewidth',2);
                set(h(k).sd,'ZData',[1,1]*-1)
            end
        end


        %Plot mean and SEM as a line, the means, and optionally the medians
        if strcmp(style,'line')
            for k=1:length(X)

                h(k).mu=plot(X(k),mu(k),'o','color','r',...
                    'markerfacecolor','r',...
                    'markersize',10);

                h(k).sem=plot([X(k),X(k)],[mu(k)-SEM(k),mu(k)+SEM(k)],'-r',...
                    'linewidth',2);
                if markMedian
                    h(k).med=plot(X(k),med(k),'s','color',[0.8,0,0],...
                        'markerfacecolor','none',...
                        'lineWidth',2,...
                        'markersize',12);
                end

                h(k).xAxisLocation=x(k);
            end
        end % if strcmp(style,'line')

        for thisInterval=1:length(h)
            h(thisInterval).interval=interval;
        end



        function ptch=patchMaker(thisInterval,tColor)
            %This nested function builds a patch for the SD or SEM
            l=mu(k)-thisInterval;
            u=mu(k)+thisInterval;
            ptch=patch([X(k)-jitScale, X(k)+jitScale, X(k)+jitScale, X(k)-jitScale],...
                [l,l,u,u], 0);
            set(ptch,'edgecolor',tColor*0.8,'facecolor',tColor)
        end %function patchMaker


        function X=violaPoints(X,Y)
            % Variable jitter according to how many points occupy each range of values.
            [counts,~,bins] = histcounts(Y,10);
            inds = find(counts~=0);
            counts = counts(inds);

            Xr = X;
            for jj=1:length(inds)
                tWidth = jitter * (1-exp(-0.1 * (counts(jj)-1)));
                xpoints = linspace(-tWidth*0.8, tWidth*0.8, counts(jj));
                Xr(bins==inds(jj)) = xpoints;
            end
            X = X+Xr;
        end % function violaPoints


    end % function myPlotter

%--- Local functions for statistics ---

    function sem=SEM_calc(vect, CI)
        % SEM_calc - standard error of the mean, confidence interval
        if isvector(vect)
            vect=vect(:);
        end

        % Define an anonymous function to take over from norminv, which is in the Stats ToolBox
        myNormInv = @(x) -sqrt(2)*erfcinv(2*x);

        if nargin==1
            stdCI = 1.96 ;
        elseif nargin==2
            CI = CI/2 ; %Convert to 2-tail
            stdCI = abs(myNormInv(CI));
        end

        for ii=1:size(vect,2)
            f =  find(~isnan(vect(:,ii)));
            sem(ii) = ( std(vect(f,ii))  ./ sqrt(length(f)) ) * stdCI ;
        end
    end

    function tint=tInterval_calc(vect, CI)
        % tInterval_calc - confidence interval based on the t-distribution
        if isvector(vect)
            vect=vect(:);
        end

        if nargin==1
            CI = 0.025; %If no second argument, work out a 2-tailed 5% t-interval
            stdCI=tinv(1-CI, length(vect)-1);
        elseif nargin==2
            CI = CI/2 ; %Convert to 2-tail
            stdCI=tinv(1-CI, length(vect)-1); %Based on the t distribution
        end

        if stdCI==0
            error('Can''t find confidence iterval for 0 standard deviations!')
        end

        for ii=1:size(vect,2)
            f =  find(~isnan(vect(:,ii)));
            tint(ii) = ( std(vect(f,ii))  ./ sqrt(length(f)) ) * stdCI ;
        end
    end

end %function notBoxPlot
