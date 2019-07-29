#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
  bcosliteclientpy is a python client for FISCO BCOS2.0 (https://github.com/FISCO-BCOS/)
  bcosliteclientpy is free software: you can redistribute it and/or modify it under the
  terms of the MIT License as published by the Free Software Foundation. This project is
  distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Thanks for
  authors and contributors of eth-abi, eth-account, eth-hash，eth-keys, eth-typing, eth-utils,
  rlp, eth-rlp , hexbytes ... and relative projects
  @file: consensus_precompile.py
  @function:
  @author: yujiechen
  @date: 2019-07
'''
import os
import json
from solc import compile_files
import subprocess
from client.common import common
from client_config import client_config
from client.bcoserror import CompilerNotFound, CompileError


class Compiler:
    """
    compile sol into bin and abi
    """
    _env_key_ = "SOLC_BINARY"
    compiler_path = client_config.solc_path
    js_compiler_path = client_config.solcjs_path
    os.putenv(_env_key_, compiler_path)
    os.environ[_env_key_] = compiler_path

    @staticmethod
    def save_file(content, file_dir, file_name):
        """
        save content to the given file
        """
        if os.path.exists(file_dir) is False:
            os.mkdir(file_dir)
        file_path = file_dir + "/" + file_name
        fp = open(file_path, 'w')
        fp.write(content)
        fp.close()

    @staticmethod
    def compile_with_js(sol_path, contract_name, output_path="contracts"):
        """
        compile with nodejs compiler
        """
        print("INFO >> compile with nodejs compiler")
        command = "{} --bin --abi {} -o {}".format(Compiler.js_compiler_path, sol_path, output_path)
        common.execute_cmd(command)
        # get oupput_prefix
        output_list = output_path.split('/')
        output_prefix = ""
        for field in output_list:
            output_prefix = output_prefix + field + "_"
        # get gen path
        gen_path = "{}/{}{}_sol_{}".format(output_path, output_prefix, contract_name, contract_name)
        target_path = "{}/{}".format(output_path, contract_name)
        bin_file = gen_path + ".bin"
        target_bin_file = target_path + ".bin"
        if os.path.exists(bin_file):
            command = "mv {} {}".format(bin_file, target_bin_file)
            common.execute_cmd(command)

        abi_file = gen_path + ".abi"
        target_abi_file = target_path + ".abi"
        if os.path.exists(abi_file):
            command = "mv {} {}".format(abi_file, target_abi_file)
            common.execute_cmd(command)

    @staticmethod
    def compile_with_solc(sol_file, contract_name, output_path):
        """
        compile with solc
        """
        print("INFO >> compile with solc compiler")
        sol_objs = compile_files([sol_file])
        key = "{}:{}".format(sol_file, contract_name)
        if key in sol_objs.keys():
            # parse abi
            if "abi" in sol_objs[key].keys():
                sol_abi = sol_objs[key]["abi"]
                # save abi
                Compiler.save_file(json.dumps(sol_abi), output_path, contract_name + ".abi")
            # parse bin
            if "bin" in sol_objs[key].keys():
                sol_bin = sol_objs[key]["bin"]
                # save bin
                if sol_bin != "":
                    Compiler.save_file(sol_bin, output_path, contract_name + ".bin")

    @staticmethod
    def compile_file(sol_file, output_path="contracts"):
        """
        get abi and bin
        """
        # get contract name
        contract_name = os.path.basename(sol_file).split('.')[0]
        try:
            # compiler not found
            if os.path.isfile(Compiler.compiler_path) is False and \
                    os.path.isfile(Compiler.js_compiler_path) is False:
                raise CompilerNotFound(("solc compiler: {} and solcjs compiler {}"
                                        " both doesn't exist,"
                                        " please install firstly !").
                                       format(Compiler.compiler_path,
                                              Compiler.js_compiler_path))
            # compile with solc if solc compiler exists
            if os.path.isfile(Compiler.compiler_path) is True:
                Compiler.compile_with_solc(sol_file, contract_name, output_path)
            # compiler with js compiler if solc compiler doesn't exist
            elif os.path.isfile(Compiler.js_compiler_path) is True:
                Compiler.compile_with_js(sol_file, contract_name, output_path)
        except CompilerNotFound as e:
            abi_path = output_path + "/" + contract_name + ".abi"
            bin_path = output_path + "/" + contract_name + ".bin"
            # exist abi
            if os.path.exists(abi_path) is False or os.path.exists(bin_path) is True:
                raise CompileError(("compile failed ! both the compiler not"
                                    " found and the bin/abi"
                                    " not found, error information: {}").format(e))
        except subprocess.CalledProcessError as e:
            raise CompileError("compile error for compile failed, error information: {}".format(e))
        except Exception as e:
            raise CompileError("compile {} failed, error information: {}".format(sol_file, e))
