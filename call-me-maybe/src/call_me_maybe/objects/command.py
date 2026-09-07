from utils import (
    print_to_stderr,
    softmax,
    decoding_strategy,
    extract_last_position_if_needed
)
from pydantic import BaseModel, Field, model_validator, ConfigDict
from .prompts import Prompt
from ..parcer import Parcer
from ..contained_decoding.function_calling import (
    FunctionCallResult,
    parse_function_call,
    validate_parameter_types,
    normalize_parameter_types,
    normalize_substitution_regex
)
from ..contained_decoding.json_validator import (
    BasicJsonFSM,
    apply_json_mask,
    load_vocab_from_model,
)
# from .generator import Generator
from ..values import Values
# from ..contained_decoding import BasicJsonFSM, load_vocab_from_model
from .function_definition import Function_definition
from llm_sdk import Small_LLM_Model
from pathlib import Path
import numpy as np


class Command(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    arguments: list[str] = Field(default_factory=list)
    llm_model: Small_LLM_Model = Field()
    project_values: Values = Field()
    response: str = Field(default="")
    prompts_list: list[Prompt] = Field(default_factory=list)
    func_definition_list: list[Function_definition] = Field(
        default_factory=list)
    function_definition_filepath: str = Field(
        default="data/input/functions_definition.json")
    input_filepath: str = Field(
        default="data/input/function_calling_tests.json")
    output_filepath: str = Field(
        default="data/output/function_calling_results.json")

    @model_validator(mode="after")
    def validate_files(self) -> "Command":
        # Change files to the Custom ones
        self.change_default_filepaths()

        if not self.do_files_exist():
            exit()

        self.create_output_file()

        self.ingest_prompts()

        self.ingest_function_definitions()

        return self

    def do_files_exist(self) -> bool:
        """Validate if files exist"""
        flag1 = not Path(self.function_definition_filepath).exists()
        flag2 = not Path(self.input_filepath).exists()

        if flag1:
            print_to_stderr(
                f"[ERROR] File '{self.function_definition_filepath}' "
                f"does not exist.")

        if flag2:
            print_to_stderr(
                f"[ERROR] File '{self.input_filepath}' "
                f"does not exist.")
        if flag1 and flag2:
            print_to_stderr(
                "[SYSTEM] Missing Function_Defenitions and Input Files.")
            return False
        elif flag1:
            print_to_stderr("[SYSTEM] Missing Function_Defenitions File.")
            return False
        elif flag2:
            print_to_stderr("[SYSTEM] Missing Input File.")
            return False

        print(
            "[SYSTEM] Function_Definitions File Exist.\n"
            "[SYSTEM] Input File Exist.\n")
        return True

    def create_output_file(self) -> None:
        """Create the Output Folder(s) and File"""
        folders_to_create: list[str] = []

        _, reversed_directory_path = self.output_filepath[::-1].split("/", 1)

        while reversed_directory_path.find("/") > 0:
            folder_name, reversed_directory_path = (
                reversed_directory_path.split("/", 1))

            folders_to_create.append(folder_name[::-1])

        folders_to_create.append(reversed_directory_path[::-1])

        current_path: str = ""
        for folder in reversed(folders_to_create):
            current_path += folder + "/"
            try:
                Path(current_path).mkdir()
                print(
                    f"[SYSTEM] Directory '{current_path}' "
                    f"created successfully.")
            except FileExistsError:
                print_to_stderr(
                    f"[WARNING] Directory '{current_path}' already exists.")
            except PermissionError:
                print_to_stderr(
                    f"[ERROR] Permission denied: "
                    f"Unable to create '{current_path}'.")
            except Exception as e:
                print_to_stderr(
                    f"[ERROR] An error occurred: {e}")

        with open(self.output_filepath, "a") as f:
            f.truncate()
            print("[SYSTEM] Output File Created.")

    def change_default_filepaths(self):
        """Change the default filepath to custum ones."""
        for arg in self.arguments:
            _, filepath = arg.split()
            if arg.startswith("--functions_definition"):
                self.function_definition_filepath = filepath
            elif arg.startswith("--input"):
                self.input_filepath = filepath
            elif arg.startswith("--output"):
                self.output_filepath = filepath
        if (
            self.function_definition_filepath == self.input_filepath
            or self.function_definition_filepath == self.output_filepath
            or self.input_filepath == self.output_filepath
        ):
            print_to_stderr(
                "[ERROR] Same file location used for multiple inputted files.")
            exit()

    def ingest_prompts(self) -> None:
        data = Parcer.load_json_safely(self.input_filepath, default=[])
        if not isinstance(data, list):
            raise ValueError(
                f"[ERROR] Expected JSON array in {self.input_filepath}")

        for item in data:
            self.prompts_list.append(Prompt(prompt=item["prompt"]))

    def ingest_function_definitions(self) -> None:
        data = Parcer.load_json_safely(
            self.function_definition_filepath,
            default=[])

        if not isinstance(data, list):
            raise ValueError(
                f"[ERROR] Expected JSON array in "
                f"{self.function_definition_filepath}")

        for func_def in data:
            parameters_list: list[tuple[str, str]] = []
            parameters_list = Parcer.get_parameters_list(func_def)
            func_def = Function_definition(
                name=func_def['name'],
                description=func_def['description'],
                parameters=parameters_list,
                returns=func_def['returns']['type'])

            self.func_definition_list.append(func_def)

    def run(self) -> None:
        """Generate and validate one function call per user prompt."""
        import json

        all_calls = []
        vocab = load_vocab_from_model(self.llm_model)

        for prompt_item in self.prompts_list:
            generation_prompt = FunctionCallResult.build_generation_prompt(
                prompt_text=prompt_item.prompt,
                definitions=self.func_definition_list,
            )

            # print(f"\n{generation_prompt}\n")

            encoded_prompt_tensor = self.llm_model.encode(
                generation_prompt
            )

            context_ids: list[int] = [
                int(token_id)
                for token_id in encoded_prompt_tensor.flatten().tolist()
            ]
            json_fsm = BasicJsonFSM(vocab)

            generated_ids: list[int] = []
            decoded_text = ""
            function_call = None

            for _ in range(self.project_values.max_tries):
                logits = self.llm_model.get_logits_from_input_ids(
                    context_ids
                )

                next_token_logits = extract_last_position_if_needed(
                    logits
                ).copy()

                banned_token_ids = getattr(
                    self.project_values,
                    "banned_token_ids",
                    [],
                )

                if banned_token_ids:
                    next_token_logits[banned_token_ids] = -np.inf

                next_token_logits = apply_json_mask(
                    next_token_logits,
                    json_fsm,
                )

                finite_token_ids = np.flatnonzero(
                    np.isfinite(next_token_logits)
                )

                if len(finite_token_ids) == 0:
                    raise ValueError(
                        "JSON mask removed every valid token. "
                        f"FSM state: {json_fsm.current_state}"
                    )

                temperature = max(
                    float(
                        getattr(
                            self.project_values,
                            "temperature",
                            1.0,
                        )
                    ),
                    1e-6,
                )

                probabilities = softmax(
                    next_token_logits,
                    temperature=temperature,
                )

                if (
                    probabilities is None
                    or not np.isfinite(probabilities).all()
                    or probabilities.sum() <= 0
                ):
                    raise ValueError(
                        "The model returned an invalid "
                        "probability distribution."
                    )

                next_token_id = int(
                    decoding_strategy(probabilities)
                )

                json_fsm.update_state(next_token_id)

                context_ids.append(next_token_id)
                generated_ids.append(next_token_id)

                if json_fsm.is_done():
                    decoded_text = self.llm_model.decode(
                        generated_ids
                    ).strip()
                    function_call = parse_function_call(decoded_text)
                    break

                # TODO Add VISUALIZATION HERE!!
                # print(f"DECODED: {decoded_text!r}")

            if function_call is None:
                print(f"RAW RESPONSE: {decoded_text!r}")
                print(
                    f"GENERATED TOKEN COUNT: {len(generated_ids)}"
                )
                print(
                    f"MAX TRIES: {self.project_values.max_tries}"
                )

                raise ValueError(
                    "Could not generate a complete JSON function call "
                    f"for prompt: {prompt_item.prompt}"
                )

            function_call.prompt = prompt_item.prompt
            normalize_substitution_regex(function_call)

            valid_function_names = {
                definition.name
                for definition in self.func_definition_list
            }

            normalize_parameter_types(
                function_call,
                self.func_definition_list,
            )

            validate_parameter_types(
                function_call,
                self.func_definition_list,
            )

            if function_call.name not in valid_function_names:
                raise ValueError(
                    f"Unknown function '{function_call.name}' returned "
                    f"for prompt '{prompt_item.prompt}'."
                )

            all_calls.append(function_call.model_dump())

        self.response = json.dumps(
            all_calls,
            indent=2,
        )

        print(
            f"generated_function_call_count = {len(all_calls)}"
        )
        print(f"response: {self.response}")

        with open(self.output_filepath, "w") as output_file:
            output_file.write(self.response)
