function[]=tlwplot(Lupper,Llower,U,H,a,ho,xdom,zdom,mink,maxk);
%-------------------------------------------------------------
% Matlab Subroutine tlwplot.M					
% Called from tlwmenu.M--interactive menu system.		
%									
% By Robert Hart for Meteo 574 / Fall 1995			
% Penn State University Meteorology
% Updated 1 March 2018 to increase npts given faster cpus
%									
% Program which analyzes and simulates the flow over			
% an isolated mountain in the presence of a two-			
% layer atmosphere.							
%									
% Parameters received from menu:					
%-------------------------------					
% Lupper - Scorer parameter of upper layer				
% Llower - Scorer parameter of lower layer				
% U      - Surface wind speed (in m/s)					
% H      - Height above ground of 2-layer interface (in m)		
% a      - Half-width of mountain (in m)				
% ho     - maximum height of mountain					
% xdom   - Horizontal extent of domain (m)				
% zdom   - Vertical extent of domain (m)				
% mink   - Minimum wavenumber in Fourier analysis of flow		
% maxk   - Maximum wavenumber in Fourier analysis of flow		
%									
%--------------------------------------------------------------

npts=100;                            % # cells in each direction
dk=0.367/a;                          % wavenumber interval size 
                                     %   (smaller the better, but
                                     %   need to watch calc. time!)  
nk=(maxk-mink)/dk;                   % # loops for wave# integration                   

minx=-.25*xdom;                      % leftmost limit of domain
maxx=.75*xdom;                       % rightmost limit of domain
minz=0;                              % lower limit of domain
maxz=zdom;                           % upper limit of domain

matrix1=zeros(npts+1,npts+1);        % temp matrix used in integration
matrix2=zeros(npts+1,npts+1);        % temp matrix used in integration
matrix3=zeros(npts+1,npts+1);        % sum matrix used in integration

dx=(maxx-minx)/npts;                 % grid cell size in horizontal
dz=(maxz-minz)/npts;                 % grid cell size in vertical

x=[minx:dx:maxx];                    % array of x gridpoints
z=[minz:dz:maxz];                    % array of z gridpoints
k=[mink:dk:maxk];                    % array of wavenumbers
           
                                     % mesh arrays into 2-d matrix
[x,z]=meshgrid(minx:dx:maxx,minz:dz:maxz); 

ht=0;                          	    % initialize weighting for transform
for kloop=1:nk;                      % wavenumber loop
  kk=k(kloop);                       %    horiz wavenumber
  m=sqrt(Llower*Llower-kk*kk);       %    vert wave# in lower layer
  n=sqrt(kk*kk-Lupper*Lupper);       %    vert wave# in upper layer
  if (m+i*n==0)                      %    check for divide by zero
    r=9e99;                           
  else 
    r=(m-i*n)/(m+i*n);               %    reflection coefficient
  end; 
  R=r*exp(2*i*m*H);                  %    reflection calculation
  A=(1+r)*exp(H*n+i*H*m)/(1+R);      %    calculate coefficients
  C=1/(1+R);                         
  D=R.*C;
  hs=pi*a*ho*exp(-a*abs(kk));        %    kk-component of Fourier trans.
  ht=ht+pi*dk*a*exp(-a*abs(kk));     %    sum Fourier weighting. if
                                     %    integrating over 0<k<oo
                                     %    ht will end up as pi.
                                     %    hence, ht represents the 
                                     %    fraction, in radians, of
                                     %    the spectral interval considered. 
                                            
  aboveH=A*exp(-z.*n).*(z>H);        %    calculate w in each layer         
  belowH=(C*exp(i*z.*m)+D*exp(-i*z.*m)).*(z<=H);
  
                                     %    combine layers, and multiply by
                                     %    transform of kk-component of topo.   
  matrix2=((-i*kk*hs*U*(aboveH+belowH)).*exp(-i*x.*kk));
   
  if kloop > 1                       %    trapezoidal integration/summation
    matrix3=matrix3+.5*(matrix1+matrix2).*dk;
  end;
 
  matrix1=matrix2;                   %    current w matrix becomes previous
end;                                 % end of integration loop.

w=real(matrix3./ht);                 % only interested in real comp. 
                                     
figure;                              % create new figure window                     
axis([minx maxx minz maxz]);         % define axes
grid;                                % graw grid
xlabel('X (m)');                     % X-axis title
ylabel('Height (m)');                % Z-axis title
title(['Streamline Analysis']);      % figure title
stream(x,z,U,w,10);                  % perform streamline analysis
Hline=H*ones(1,npts+1);              % create interface line        
plot(x(1,:),Hline,'m--');            % draw interface line in magenta!

figure;                              % create new figure window                                                      
colormap(jet);                       % set colormap (red=high;blue=low)
axis([minx maxx minz maxz]);         % define axes
xlabel('X (m)');                     % X-axis title
ylabel('Height (m)');                % Z-axis title
title(['Vertical Velocity (m/s)']);  % figure title
hold on;                             % hold current figure
plot(x(1,:),Hline,'m--');            % draw interface line
caxis([-10 10]);                     % define color contouring extrema
surface(x,z,w);                      % 2-d surface contour plot
shading interp;                      % interpolate between gridpoints
colorbar;                            % add contour legend
hold off;                            % remove hold on figure
