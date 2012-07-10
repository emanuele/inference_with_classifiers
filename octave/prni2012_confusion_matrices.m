function PRNI2012_CMs = prni2012_confusion_matrices()

%%% Presentation at PRNI 2012
%%% Testing Multiclass Pattern Discrimination
%%% p.11

% Discrimination of all classes A, B, and C.
PRNI2012_CMs.pred_all = [
    40 10 10;
    10 40 10;
    10 10 40];

% No discrimination between A and B, but between [A,B] and C.
PRNI2012_CMs.pred_group = [
     30 30 0;
     30 30 0;
     0 0 60];
 
% No discrimination between A, B, and C
PRNI2012_CMs.pred_random = [
    20 20 20;
    20 20 20;
    20 20 20];

end