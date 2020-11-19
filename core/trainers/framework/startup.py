# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import warnings
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddlerec.core.utils import envs

__all__ = [
    "StartupBase", "SingleStartup", "PSStartup", "CollectiveStartup",
    "FineTuningStartup"
]


class StartupBase(object):
    """R
    """

    def __init__(self, context):
        pass

    def startup(self, context):
        pass

    def load(self, context, is_fleet=False, main_program=None):
        dirname = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path", None)
        if dirname is None or dirname == "":
            return
        print("going to load ", dirname)
        fluid.io.load_persistables(
            context["exe"], dirname, main_program=main_program)
        print("load from {} success".format(dirname))


class SingleStartup(StartupBase):
    """R
    """

    def __init__(self, context):
        print("Running SingleStartup.")

    def startup(self, context):
        for model_dict in context["phases"]:
            with paddle.static.scope_guard(context["model"][model_dict["name"]]
                                           ["scope"]):
                train_prog = context["model"][model_dict["name"]][
                    "main_program"]
                startup_prog = context["model"][model_dict["name"]][
                    "startup_program"]
                with paddle.static.program_guard(train_prog, startup_prog):
                    context["exe"].run(startup_prog)
                    self.load(context, main_program=train_prog)
        context["status"] = "train_pass"


class FineTuningStartup(StartupBase):
    """R
    """

    def __init__(self, context):
        self.op_name_scope = "op_namescope"
        self.clip_op_name_scope = "@CLIP"
        self.self.op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName(
        )

        print("Running SingleStartup.")

    def _is_opt_role_op(self, op):
        # NOTE: depend on oprole to find out whether this op is for
        # optimize
        op_maker = core.op_proto_and_checker_maker
        optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
        if op_maker.kOpRoleAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize_role):
            return True
        return False

    def _get_params_grads(self, program):
        """
        Get optimizer operators, parameters and gradients from origin_program
        Returns:
            opt_ops (list): optimize operators.
            params_grads (dict): parameter->gradient.
        """
        block = program.global_block()
        params_grads = []
        # tmp set to dedup
        optimize_params = set()
        origin_var_dict = program.global_block().vars
        for op in block.ops:
            if self._is_opt_role_op(op):
                # Todo(chengmo): Whether clip related op belongs to Optimize guard should be discussed
                # delete clip op from opt_ops when run in Parameter Server mode
                if self.op_name_scope in op.all_attrs(
                ) and self.clip_op_name_scope in op.attr(self.op_name_scope):
                    op._set_attr(
                        "op_role",
                        int(core.op_proto_and_checker_maker.OpRole.Backward))
                    continue

                if op.attr(self.op_role_var_attr_name):
                    param_name = op.attr(self.op_role_var_attr_name)[0]
                    grad_name = op.attr(self.op_role_var_attr_name)[1]
                    if not param_name in optimize_params:
                        optimize_params.add(param_name)
                        params_grads.append([
                            origin_var_dict[param_name],
                            origin_var_dict[grad_name]
                        ])
        return params_grads

    @staticmethod
    def is_persistable(var):
        """
        Check whether the given variable is persistable.

        Args:
            var(Variable): The variable to be checked.

        Returns:
            bool: True if the given `var` is persistable
            False if not.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                param = paddle.static.default_main_program().global_block().var('fc.b')
                res = fluid.io.is_persistable(param)
        """
        if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                var.desc.type() == core.VarDesc.VarType.READER:
            return False
        return var.persistable

    def load(self, context, is_fleet=False, main_program=None):
        dirname = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path", None)
        if dirname is None or dirname == "":
            return
        print("going to load ", dirname)

        params_grads = self._get_params_grads(main_program)
        update_params = [p for p, _ in params_grads]
        need_load_vars = []
        parameters = list(
            filter(FineTuningStartup.is_persistable, main_program.list_vars()))

        for param in parameters:
            if param not in update_params:
                need_load_vars.append(param)

        fluid.io.load_vars(context["exe"], dirname, main_program,
                           need_load_vars)
        print("load from {} success".format(dirname))

    def startup(self, context):
        for model_dict in context["phases"]:
            with paddle.static.scope_guard(context["model"][model_dict["name"]]
                                           ["scope"]):
                train_prog = context["model"][model_dict["name"]][
                    "main_program"]
                startup_prog = context["model"][model_dict["name"]][
                    "startup_program"]
                with paddle.static.program_guard(train_prog, startup_prog):
                    context["exe"].run(startup_prog)
                    self.load(context, main_program=train_prog)
        context["status"] = "train_pass"


class FleetStartup(StartupBase):
    def __init__(self, context):
        print("Running FleetStartup.")
        pass

    def startup(self, context):
        model_dict = context["env"]["phase"][0]
        with paddle.static.scope_guard(context["model"][model_dict["name"]][
                "scope"]):
            train_prog = context["model"][model_dict["name"]][
                "default_main_program"]
            startup_prog = context["model"][model_dict["name"]][
                "startup_program"]
            with paddle.static.program_guard(train_prog, startup_prog):
                context["exe"].run(startup_prog)
                if context['fleet'].is_worker():
                    # for parameter-server worker
                    context["fleet"].init_worker()
                else:
                    # for collective
                    self.load(context, main_program=train_prog)
        context["status"] = "train_pass"


class SingleInferStartup(StartupBase):
    def __init__(self, context):
        print("Running SingleInferStartup.")
        pass

    def startup(self, context):
        for model_dict in context["phases"]:
            with paddle.static.scope_guard(context["model"][model_dict["name"]]
                                           ["scope"]):
                train_prog = context["model"][model_dict["name"]][
                    "main_program"]
                startup_prog = context["model"][model_dict["name"]][
                    "startup_program"]
                with paddle.static.program_guard(train_prog, startup_prog):
                    context["exe"].run(startup_prog)
        context["status"] = "train_pass"
