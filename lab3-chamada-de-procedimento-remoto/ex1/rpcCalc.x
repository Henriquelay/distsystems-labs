struct operands {
       int left;
       int right; 
};
program PROG { 
       version VER { 
               int ADD(operands) = 1; 
               int SUB(operands) = 2; 
       } = 100;
} = 5555;
