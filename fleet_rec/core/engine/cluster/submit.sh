#!/bin/bash

###################################################
# Usage: submit.sh
# Description: run mpi submit clinet
###################################################

# ---------------------------------------------------------------------------- #
#                            variable define                                   #
# ---------------------------------------------------------------------------- #
declare g_curPath=""
declare g_scriptName=""
declare g_workPath=""
declare g_run_stage=""

# ---------------------------------------------------------------------------- #
#                             const define                                     #
# ---------------------------------------------------------------------------- #
declare -r CALL="x"
################################################################################


#-----------------------------------------------------------------------------------------------------------------
# Function: get_cur_path
# Description: get churrent path
# Parameter:
#   input:
#   N/A
#   output:
#   N/A
# Return: 0 -- success; not 0 -- failure
# Others: N/A
#-----------------------------------------------------------------------------------------------------------------
get_cur_path()
{
  g_run_stage="get_cur_path"
    cd "$(dirname "${BASH_SOURCE-$0}")"
    g_curPath="${PWD}"
    g_scriptName="$(basename "${BASH_SOURCE-$0}")"
    cd - >/dev/null
}

#-----------------------------------------------------------------------------------------------------------------
#fun : check function return code
#param : N/A
#return : 0 -- success; not 0 -- failure
#-----------------------------------------------------------------------------------------------------------------
function check_error()
{
    if [ ${?} -ne 0 ]
    then
        echo "execute " + $g_run_stage +  " raise exception! please check ..."
        exit 1
    fi
}
