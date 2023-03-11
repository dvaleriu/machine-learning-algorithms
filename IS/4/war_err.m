%
%	File WAR_ERR.M
%
%	Function: WAR_ERR
%
%	Call: war_err(msg)
%
%	Displays a warning/error message. If msg is 
%	present and non void, then it is displayed. 
%	Otherwise (missing or void), a standard error 
%	message is displayed. 
%

function war_err(msg)

%
%	BEGIN
%
    if ((nargin > 0) && (~isempty(msg)))
	   disp(' ') ;
	   disp(msg) ;
	   disp(' ') ;  
    else
	   disp(' ') ;
	   disp('<WAR_ERR>: An error occured.') ;
	   disp(' ') ;
    end
%
%
%	END
%

