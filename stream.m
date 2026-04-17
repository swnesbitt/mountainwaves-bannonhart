function[]=stream(x,y,U,v,num);
%-------------------------------------------------------%
% Matlab subroutine stream.m				%
% Called by:  tlwplot.m					%
%							%
% by Robert Hart for Meteo 574 / Fall 95		%
% Penn State University Meteorology                     %
%							%
% This subroutine performs a streamline analysis	%
% of the wind field.  					%
%							%
% Parameters passed to this routine:			%
% ----------------------------------			%
% x	- array of x-gridpoints				%
% y	- array of y-gridpoints				%
% U 	- speed of x-direction wind (constant)		%	
% v	- array of y-direction wind			%
% num	- # of evenly spaced streamlines to draw	%
%							%
% NOTE:  Subroutine is written for a constant   	%
% 	 horizontal wind velocity.			%
%-------------------------------------------------------%
hold;				% hold figure
xsize=size(x);				
xsize=xsize(1);			% size of x-grid
ysize=size(y);
ysize=ysize(2);                 % size of y-grid
miny=y(1);                      % min. y-value                  
maxy=y(ysize);                  % max. y-value
minx=x(1);                      % min. x-value
maxx=x(1,xsize);                % max. x-value

dx=(maxx-minx)/xsize;           % x-dir gridspacing
dy=(maxy-miny)/ysize;           % y-dir grid spacing
dh=ysize/num;                   % streamline spacing

tstep=dx/U;                     % time to cross cell

mtncolor=[.02 .77 .02];         % color of mountain (lt green here)

ycell=1;                        % initial location
for j=1:num                     % streamline loop
  ycell=1+dh*(j-1);             %   current y-location
  if ycell < 1                  %   don't go outside array
    ycell=1;
  end;
  if ycell > ysize              %   don't go outside array
     ycell=ysize;
  end;
  ax=[];                        %   initialize lineplot arrays
  ay=[];               
  ax(1)=minx;                   
  ay(1)=y(round(ycell));
  ycell=int8(ycell);
  for i=2:xsize;                %   streamline tracing loop
     ax(i)=x(ycell,i);          %      calculate displacement
     ay(i)=ay(i-1)+tstep*v(ycell,i);
  end;                          %   end of loop
  if j==1                       %   if first streamline                    
     ax(xsize+1)=maxx;
     ay(ysize+1)=miny;
     fill(ax,ay,mtncolor);      %      solid-fill mountain
  else                          %   otherwise,
     line(ax,ay);	        %      draw streamline!
  end;                          %   end of loop
end;                            % the end;
