import gradio as gr
import pandas as pd
import json
import os
import random
from dotenv import load_dotenv

load_dotenv()

from core import compile_program, list_prompts, export_to_csv, generate_program_response


# Gradio interface
custom_css = """
.expand-button {
  min-width: 20px !important;
  width: 20px !important;
  padding: 0 !important;
  font-size: 10px !important;
}
.prompt-card {
  height: 150px !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  padding: 10px !important;
  position: relative !important;
}
.prompt-details {
  flex-grow: 1 !important;
}
.view-details-btn {
  position: absolute !important;
  bottom: 10px !important;
  right: 10px !important;
}
.red-text {
  color: red !important;
}
"""

example2_signature = "JokeTopic:Funny-Gpt4oMini_ChainOfThought_Bootstrapfewshotwithrandomsearch-20241003.json - joke, topic -> funny (Score: 100)"

with gr.Blocks(css=custom_css) as demo:

    # Compile Program Tab
    with gr.Tabs():
        with gr.TabItem("编译程序"):

            with gr.Row():
                with gr.Column():
                    gr.Markdown("# DSPyUI: DSPy 的 Gradio 用户界面")
                    gr.Markdown("通过指定设置并提供示例数据来构建 DSPy 程序。")

                with gr.Column():
                    gr.Markdown("### 示例演示：")
                    with gr.Row():  
                        example1 = gr.Button("笑话评分")
                        example2 = gr.Button("讲笑话")
                        example3 = gr.Button("改写笑话")
            
            # Task Instructions
            with gr.Row():
                with gr.Column(scale=4):
                    instructions = gr.Textbox(
                        label="任务指令",
                        lines=3,
                        placeholder="在此输入详细的任务指令。",
                        info="为任务提供清晰全面的指令。这将引导 DSPy 程序理解具体要求和预期结果。",
                        interactive=True  # Add this line to ensure the textbox is editable
                    )

            input_values = gr.State([])
            output_values = gr.State([])
            file_data = gr.State(None)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 输入 (Inputs)")
                    gr.Markdown("为您的任务添加输入字段。每个输入字段代表 DSPy 程序将接收的一条信息。")
                    with gr.Row():
                        add_input_btn = gr.Button("添加输入字段")
                        remove_input_btn = gr.Button("移除最后一个输入", interactive=False)

                with gr.Column():
                    gr.Markdown("### 输出 (Outputs)")
                    gr.Markdown("为您的任务添加输出字段。每个输出字段代表 DSPy 程序将生成的一条信息。")
                    with gr.Row():  
                        add_output_btn = gr.Button("添加输出字段")
                        remove_output_btn = gr.Button("移除最后一个输出", interactive=False)

            def add_field(values):
                new_values = values + [("", "")]
                return new_values, gr.update(interactive=True)

            def remove_last_field(values):
                new_values = values[:-1] if values else values
                return new_values, gr.update(interactive=bool(new_values))

            add_input_btn.click(
                add_field,
                inputs=input_values,
                outputs=[input_values, remove_input_btn]
            )

            remove_input_btn.click(
                remove_last_field,
                inputs=input_values,
                outputs=[input_values, remove_input_btn]
            )

            add_output_btn.click(
                add_field,
                inputs=output_values,
                outputs=[output_values, remove_output_btn]
            )

            remove_output_btn.click(
                remove_last_field,
                inputs=output_values,
                outputs=[output_values, remove_output_btn]
            )

            def load_csv(filename):
                try:
                    df = pd.read_csv(f"example_data/{filename}")
                    return df
                except Exception as e:
                    print(f"Error loading CSV: {e}")
                    return None
                
            row_choice_options = gr.State([])

            @gr.render(inputs=[input_values, output_values, file_data])
            def render_variables(input_values, output_values, file_data):
                
                inputs = []
                outputs = []
                with gr.Row():
                    with gr.Column():
                        if not input_values:
                            gr.Markdown("请添加至少一个输入字段。", elem_classes="red-text")
   
                        for i, input_value in enumerate(input_values):
                            name, desc = input_value
                            with gr.Group():
                                with gr.Row():
                                    input_name = gr.Textbox(
                                        placeholder=f"输入{i+1}",
                                        value=name if name else None,
                                        key=f"input-name-{i}",
                                        show_label=False,
                                        label=f"输入 {i+1} 名称",
                                        info="指定此输入字段的名称。",
                                        interactive=True,
                                        scale=9
                                    )
                                    expand_btn = gr.Button("▼", size="sm", scale=1, elem_classes="expand-button")
                                input_desc = gr.Textbox(
                                    value=desc if desc else None,
                                    placeholder=desc if desc else "描述 (可选)",
                                    key=f"input-desc-{i}",
                                    show_label=False,
                                    label=f"输入 {i+1} 描述",
                                    info="（可选）为此输入字段提供描述。",
                                    interactive=True,
                                    visible=False
                                )
                                desc_visible = gr.State(False)
                                expand_btn.click(
                                    lambda v: (not v, gr.update(visible=not v)),
                                    inputs=[desc_visible],
                                    outputs=[desc_visible, input_desc]
                                )
                                inputs.extend([input_name, input_desc, desc_visible])
                    
                    with gr.Column():
                        
                        if not output_values:
                            gr.Markdown("请添加至少一个输出字段。", elem_classes="red-text")
           
                        for i, output_value in enumerate(output_values):
                            name, desc = output_value
                            with gr.Group():
                                with gr.Row():
                                    output_name = gr.Textbox(
                                        placeholder=f"输出{i+1}",
                                        value=name if name else None,
                                        key=f"output-name-{i}",
                                        show_label=False,
                                        label=f"输出 {i+1} 名称",
                                        info="指定此输出字段的名称。",
                                        scale=9,
                                        interactive=True,
                                    )
                                    expand_btn = gr.Button("▼", size="sm", scale=1, elem_classes="expand-button")
                                output_desc = gr.Textbox(
                                    value=desc if desc else None,
                                    placeholder=desc if desc else "描述 (可选)",
                                    key=f"output-desc-{i}",
                                    show_label=False,
                                    label=f"输出 {i+1} 描述",
                                    info="（可选）为此输出字段提供描述。",
                                    visible=False,
                                    interactive=True,
                                )
                                desc_visible = gr.State(False)
                                expand_btn.click(
                                    lambda v: (not v, gr.update(visible=not v)),
                                    inputs=[desc_visible],
                                    outputs=[desc_visible, output_desc]
                                )
                                outputs.extend([output_name, output_desc, desc_visible])

                    def update_judge_prompt_visibility(metric, *args):
                        # Correctly assign input and output fields based on the actual arguments
                        input_fields = []
                        output_fields = []
                        filtered_args = [args[i] for i in range(0, len(args), 3)]  # Filter out descriptions and visibility
                        for arg in filtered_args:
                            if arg and isinstance(arg, str) and arg.strip():
                                if len(input_fields) < len(input_values):
                                    input_fields.append(arg)
                                elif len(output_fields) < len(output_values):
                                    output_fields.append(arg)

                        if metric == "LLM-as-a-Judge":
                            
                            prompts = list_prompts(output_filter=input_fields + output_fields)
                            choices = [f"{p['ID']} - {p['Signature']} (Score: {p['Eval Score']})" for p in prompts] 
                            choices.append(example2_signature)

                            return gr.update(visible=True, choices=choices, value=example2_signature)
                        else:
                            return gr.update(visible=False, choices=[])

                    metric_type.change(
                        update_judge_prompt_visibility,
                        inputs=[metric_type] + inputs + outputs,
                        outputs=[judge_prompt]
                    )
                    random_row_button.click(
                        select_random_row,
                        inputs=[row_choice_options],
                        outputs=[row_selector]
                    )

                    def compile(data):
                        input_fields = []
                        input_descs = []
                        output_fields = []
                        output_descs = []
                        
                        for i in range(0, len(inputs), 3):
                            if data[inputs[i]].strip():
                                input_fields.append(data[inputs[i]])
                                if data[inputs[i+1]].strip():
                                    input_descs.append(data[inputs[i+1]])
                        
                        for i in range(0, len(outputs), 3):
                            if data[outputs[i]].strip():
                                output_fields.append(data[outputs[i]])
                                if data[outputs[i+1]].strip():
                                    output_descs.append(data[outputs[i+1]])

                        # Get the judge prompt ID if LLM-as-a-Judge is selected
                        judge_prompt_id = None
                        if data[metric_type] == "LLM-as-a-Judge":
                            judge_prompt_id = data[judge_prompt].split(' - ')[0]

                        hint = data[hint_textbox] if data[dspy_module] == "ChainOfThoughtWithHint" else None
                        
                        usage_instructions, optimized_prompt = compile_program(
                            input_fields,
                            output_fields,
                            data[dspy_module],
                            data[llm_model], 
                            data[teacher_model],
                            data[example_data],
                            data[optimizer],
                            data[instructions],
                            data[metric_type],
                            judge_prompt_id,
                            input_descs,
                            output_descs,
                            hint  # Add the hint parameter
                        )
                        
                        signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
                        
                        # Extract evaluation score from usage_instructions
                        score_line = [line for line in usage_instructions.split('\n') if line.startswith("Evaluation score:")][0]
                        evaluation_score = float(score_line.split(":")[1].strip())
                        
                        # Remove the evaluation score line from usage_instructions
                        usage_instructions = '\n'.join([line for line in usage_instructions.split('\n') if not line.startswith("Evaluation score:")])

                        # Extract baseline score from usage_instructions
                        baseline_score_line = [line for line in usage_instructions.split('\n') if line.startswith("Baseline score:")][0]
                        baseline_score = float(baseline_score_line.split(":")[1].strip())

                        # Remove the baseline score line from usage_instructions
                        usage_instructions = '\n'.join([line for line in usage_instructions.split('\n') if not line.startswith("Baseline score:")])

                        # Extract human-readable ID from usage_instructions
                        human_readable_id = None
                        for line in usage_instructions.split('\n'):
                            if "programs/" in line and ".json" in line:
                                human_readable_id = line.split('programs/')[1].split('.json')[0]
                                break
                        
                        if human_readable_id is None:
                            raise ValueError("Could not extract human-readable ID from usage instructions")

                        # Save details to JSON
                        details = {
                            "input_fields": input_fields,
                            "input_descriptions": input_descs,
                            "output_fields": output_fields,
                            "output_descriptions": output_descs,
                            "dspy_module": data[dspy_module],
                            "llm_model": data[llm_model],
                            "teacher_model": data[teacher_model],
                            "optimizer": data[optimizer],
                            "instructions": data[instructions],
                            "signature": signature,
                            "evaluation_score": evaluation_score,
                            "baseline_score": baseline_score,
                            "optimized_prompt": optimized_prompt,
                            "usage_instructions": usage_instructions,
                            "human_readable_id": human_readable_id
                        }

                        row_choice_options = [f"Row {i+1}" for i in range(len(data[example_data]))]
                        
                        # Create 'prompts' folder if it doesn't exist
                        if not os.path.exists('prompts'):
                            os.makedirs('prompts')
                        
                        # Save JSON file with human-readable ID
                        json_filename = f"prompts/{human_readable_id}.json"
                        with open(json_filename, 'w') as f:
                            json.dump(details, f, indent=4)
                        return signature, evaluation_score, optimized_prompt, gr.update(choices=row_choice_options, visible=True, value="Row 1"), gr.update(visible=True), row_choice_options, gr.update(visible=True), gr.update(visible=True), human_readable_id, gr.update(visible=True), baseline_score
                    
                gr.Markdown("### 数据 (Data)")
                gr.Markdown("为您的任务提供示例数据。这将帮助 DSPy 编译器理解您的数据格式。您可以手动输入数据，也可以上传带有正确列标题的 CSV 文件。")
                with gr.Column():
                    with gr.Row():
                        enter_manually_btn = gr.Button("手动输入", interactive=len(input_values) > 0 and len(output_values) > 0 and file_data is None)
                        
                        upload_csv_btn = gr.UploadButton("上传 CSV", file_types=[".csv"], interactive=len(input_values) > 0 and len(output_values) > 0 and file_data is None)

                    headers = [input_value[0] for input_value in input_values] + [output_value[0] for output_value in output_values]
                        
                    example_data = gr.Dataframe(
                        headers=headers,
                        datatype=["str"] * (len(input_values) + len(output_values)),
                        interactive=True,
                        row_count=1,
                        col_count=(len(input_values) + len(output_values), "fixed"),
                        visible=file_data is not None,  # Only visible if file_data is not None
                        label="示例数据",
                        value=file_data if file_data is not None else pd.DataFrame(columns=headers)
                    )
                    export_csv_btn = gr.Button("导出为 CSV", interactive=file_data is not None and len(input_values) > 0 and len(output_values) > 0)
                    csv_download = gr.File(label="下载 CSV", visible=False)
                    error_message = gr.Markdown()
                    
                    def show_dataframe(*args):
                        # Correctly assign input and output fields based on the actual arguments
                        input_fields = []
                        output_fields = []
                        filtered_args = [args[i] for i in range(0, len(args), 3)]  # Filter out descriptions and visibility
                        input_names = [name for name, _ in input_values]
                        for arg in filtered_args:
                            if arg and isinstance(arg, str) and arg.strip():
                                if len(input_fields) < len(input_names):
                                    input_fields.append(arg)
                                elif len(output_fields) < len(output_values):
                                    output_fields.append(arg)

                        headers = input_fields + output_fields
                        
                        # Create a new dataframe with the correct headers
                        new_df = pd.DataFrame(columns=headers)
                        
                        return gr.update(visible=True, value=new_df), gr.update(visible=True), gr.update(visible=True), gr.update(interactive=False), gr.update(interactive=False)

                    enter_manually_btn.click(
                        show_dataframe,
                        inputs=inputs + outputs,
                        outputs=[example_data, export_csv_btn, compile_button, enter_manually_btn, upload_csv_btn]
                    )
                    upload_csv_btn.upload(
                        process_csv,
                        inputs=[upload_csv_btn] + inputs + outputs,
                        outputs=[example_data, example_data, compile_button, error_message, enter_manually_btn, upload_csv_btn]
                    )

                    export_csv_btn.click(
                        export_to_csv,
                        inputs=[example_data],
                        outputs=[csv_download]
                    ).then(
                        lambda: gr.update(visible=True),
                        outputs=[csv_download]
                    )
                
                compile_button.click(
                    compile,
                    inputs=set(inputs + outputs + [llm_model, teacher_model, dspy_module, example_data, upload_csv_btn, optimizer, instructions, metric_type, judge_prompt, hint_textbox]),
                    outputs=[signature, evaluation_score, optimized_prompt, row_selector, random_row_button, row_choice_options, generate_button, generate_output, human_readable_id, human_readable_id, baseline_score]
                )

                def generate_response(human_readable_id, row_selector, df):
                    selected_row = df.iloc[int(row_selector.split()[1]) - 1].to_dict()
                    print("selected_row:", selected_row)
                    return generate_program_response(human_readable_id, selected_row)

                generate_button.click(
                    generate_response,
                    inputs=[human_readable_id, row_selector, example_data],
                    outputs=[generate_output]
                )

            gr.Markdown("### 设置 (Settings)")
            with gr.Row():
                model_options = [
                    "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini",
                    "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                    "mixtral-8x7b-32768", "gemma-7b-it", "llama3-70b-8192",
                    "llama3-8b-8192", "gemma2-9b-it", "gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-3-pro-preview"
                ]
                llm_model = gr.Dropdown(
                    model_options,
                    label="模型 (Model)",
                    value="gpt-4o-mini",
                    info="选择您的 DSPy 程序的主语言模型。该模型将用于推理。通常建议选择一个快速且便宜的模型，并通过训练提高其质量。",
                    interactive=True  # Add this line
                )
                teacher_model = gr.Dropdown(
                    model_options,
                    label="教师模型 (Teacher)",
                    value="gpt-4o",
                    info="在编译过程中选择一个更强大的模型作为教师模型。该模型有助于生成高质量的示例并精炼提示词。",
                    interactive=True  # Add this line
                )
                with gr.Column():
                    dspy_module = gr.Dropdown(
                        ["Predict", "ChainOfThought", "ChainOfThoughtWithHint"],
                        label="模块 (Module)",
                        value="Predict",
                        info="选择最适合您任务的 DSPy 模块。Predict 用于简单任务，ChainOfThought 用于复杂推理，ChainOfThoughtWithHint 用于引导式推理。",
                        interactive=True  # This line was likely already present
                    )
                    hint_textbox = gr.Textbox(
                        label="提示 (Hint)",
                        lines=2,
                        placeholder="为带有 Hint 的思维链模块输入提示词。",
                        visible=False,
                        interactive=True  # Add this line
                    )

            with gr.Row():
                optimizer = gr.Dropdown(
                    ["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPRO", "MIPROv2", "COPRO"],
                    label="优化器 (Optimizer)",
                    value="BootstrapFewShot",
                    info="选择优化策略：None（不优化）；BootstrapFewShot（小数据集，约10个示例）使用少样本学习；BootstrapFewShotWithRandomSearch（中等，约50个）增加随机搜索；MIPRO, MIPROv2 和 COPRO（大型，300+）同时优化提示指令。",
                    interactive=True  # Add this line
                )
                with gr.Column():
                    metric_type = gr.Radio(
                        ["Exact Match", "Cosine Similarity", "LLM-as-a-Judge"],
                        label="评估指标 (Metric)",
                        value="Exact Match",
                        info="选择如何评估程序的性能。Exact Match 适用于有清晰正确答案的任务；LLM-as-a-Judge 更适合开放式或主观任务；Cosine Similarity 可用于输出需要与正确答案相似的模糊匹配任务。",
                        interactive=True  # Add this line
                    )
                    judge_prompt = gr.Dropdown(
                        choices=[],
                        label="评判提示词 (Judge Prompt)",
                        visible=False,
                        info="选择用作评估评判的提示词程序。",
                        interactive=True  # Add this line
                    )

            compile_button = gr.Button("编译程序", visible=False, variant="primary")
            with gr.Column() as compilation_results:
                gr.Markdown("### 编译结果 (Results)")
                
                with gr.Row():
                    signature = gr.Textbox(label="签名 (Signature)", interactive=False, info="编译后的 DSPy 程序签名，展示输入和输出字段。")
                    evaluation_score = gr.Number(label="评估分数", info="编译后的 DSPy 程序的评估得分。", interactive=False)
                    baseline_score = gr.Number(label="基准分数", info="优化前原始 DSPy 模块的基准得分。", interactive=False)
                    
                optimized_prompt = gr.Textbox(label="优化后的提示词", info="DSPy 编译器为您的程序生成的优化提示内容。", interactive=False)

            with gr.Row():

                # Add a dropdown to select a row from the dataset
                with gr.Column(scale=1):
                    human_readable_id = gr.Textbox(interactive=False, visible=False)
                    row_selector = gr.Dropdown(
                        choices=[],
                        label="从数据集中选择一行",
                        interactive=True,
                        visible=False,
                        info="从加载的数据集中选择一个特定的行，作为编译后程序的输入进行测试。"
                    )
                    random_row_button = gr.Button("随机选择一行", visible=False, interactive=True)
                    generate_button = gr.Button("生成 (Generate)", interactive=True, visible=False, variant="primary")


                    

                with gr.Column(scale=2):
                    
                    generate_output = gr.Textbox(label="生成的响应", info="由编译后的 DSPy 程序生成的输入和输出结果。", interactive=False, lines=10, visible=False)

                def select_random_row(row_choice_options):                    
                    if row_choice_options:
                        random_choice = random.choice(row_choice_options)
                        return gr.update(value=random_choice, visible=True)
                    return gr.update(visible=True)

            def process_csv(file, *args):
                if file is not None:
                    try:
                        df = pd.read_csv(file.name)
                        # Correctly assign input and output fields based on the actual arguments
                        input_fields = []
                        output_fields = []
                        filtered_args = [args[i] for i in range(0, len(args), 3)]  # Filter out descriptions and visibility
                        for arg in filtered_args:
                            if arg and isinstance(arg, str) and arg.strip():
                                if len(input_fields) < len(input_values):
                                    input_fields.append(arg)
                                elif len(output_fields) < len(output_values):
                                    output_fields.append(arg)
                        expected_headers = input_fields + output_fields
                        
                        if list(df.columns) != expected_headers:
                            return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: CSV headers do not match expected format. Expected: {expected_headers}, Got: {list(df.columns)}")
                        return df, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                    except Exception as e:
                        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: {str(e)}")
                return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False)

            # Function to show/hide the hint textbox based on the selected module
            def update_hint_visibility(module):
                return gr.update(visible=module == "ChainOfThoughtWithHint")

            # Connect the visibility update function to the module dropdown
            dspy_module.change(update_hint_visibility, inputs=[dspy_module], outputs=[hint_textbox])

            
            def disable_example_buttons():
                return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

            def update_example2():
                return (
                    gr.update(value="Tell me a funny joke"),
                    gr.update(value="MIPROv2"),
                    gr.update(value="LLM-as-a-Judge"),
                    gr.update(value="gpt-4o-mini"),
                    gr.update(value="gpt-4o"),
                    gr.update(value="Predict"),
                    [("topic", "The topic of the joke")],
                    [("joke", "The funny joke")],
                    *disable_example_buttons(),
                    load_csv("telling_jokes.csv"),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value=example2_signature, visible=True)  # Update judge_prompt
                )

            example2.click(
                update_example2,
                inputs=[],
                outputs=[instructions, optimizer, metric_type, llm_model, teacher_model, dspy_module, input_values, output_values, example1, example2, example3, file_data, compile_button, judge_prompt]
            )

            example1.click(
                lambda _: (
                    gr.update(value="Rate whether a joke is funny"),
                    gr.update(value="BootstrapFewShotWithRandomSearch"),
                    gr.update(value="Exact Match"),
                    gr.update(value="gpt-4o-mini"),
                    gr.update(value="gpt-4o"),
                    gr.update(value="ChainOfThought"),
                    [("joke", "The joke to be rated"), ("topic", "The topic of the joke")],
                    [("funny", "Whether the joke is funny or not, 1 or 0.")],
                    *disable_example_buttons(),
                    load_csv("rating_jokes.csv"),
                    gr.update(visible=True)
                ),
                inputs=[gr.State(None)],
                outputs=[instructions, optimizer, metric_type, llm_model, teacher_model, dspy_module, input_values, output_values, example1, example2, example3, file_data, compile_button]
            )

            example3.click(
                lambda _: (
                    gr.update(value="Rewrite in a comedian's style"),
                    gr.update(value="BootstrapFewShot"),
                    gr.update(value="Cosine Similarity"),
                    gr.update(value="claude-3-haiku-20240307"),
                    gr.update(value="claude-3-sonnet-20240229"),
                    gr.update(value="Predict"),
                    [("joke", "The joke to be rewritten"), ("comedian", "The comedian the joke should be rewritten in the style of")],
                    [("rewritten_joke", "The rewritten joke")],
                    *disable_example_buttons(),
                    load_csv("rewriting_jokes.csv"),
                    gr.update(visible=True)
                ),
                inputs=[gr.State(None)],
                outputs=[instructions, optimizer, metric_type, llm_model, teacher_model, dspy_module, input_values, output_values, example1, example2, example3, file_data, compile_button]
            )

            

        with gr.TabItem("查看提示词"):
            
            prompts = list_prompts()

            selected_prompt = gr.State(None)
            
            # Extract unique signatures for the dropdown
            unique_signatures = sorted(set(p["Signature"] for p in prompts))

            close_details_btn = gr.Button("关闭详情", elem_classes="close-details-btn", size="sm", visible=False)
            close_details_btn.click(lambda: (None, gr.update(visible=False)), outputs=[selected_prompt, close_details_btn])
            

            @gr.render(inputs=[selected_prompt])
            def render_prompt_details(selected_prompt):
                if selected_prompt is not None:
                    with gr.Row():
                        with gr.Column():
                            details = json.loads(selected_prompt["Details"])
                            gr.Markdown(f"## {details['human_readable_id']}")
                            with gr.Group():
                                with gr.Column(elem_classes="prompt-details-full"):
                                    gr.Number(value=float(selected_prompt['Eval Score']), label="评估分数", interactive=False)
                                    
                                    with gr.Row():
                                        gr.Dropdown(choices=details['input_fields'], value=details['input_fields'], label="输入字段", interactive=False, multiselect=True, info=", ".join(details.get('input_descs', [])))
                                        gr.Dropdown(choices=details['output_fields'], value=details['output_fields'], label="输出字段", interactive=False, multiselect=True, info=", ".join(details.get('output_descs', [])))
                                    
                                    with gr.Row():
                                        gr.Dropdown(choices=[details['dspy_module']], value=details['dspy_module'], label="模块", interactive=False)
                                        gr.Dropdown(choices=[details['llm_model']], value=details['llm_model'], label="模型", interactive=False)
                                        gr.Dropdown(choices=[details['teacher_model']], value=details['teacher_model'], label="教师模型", interactive=False)
                                        gr.Dropdown(choices=[details['optimizer']], value=details['optimizer'], label="优化器", interactive=False)
                                    
                                    gr.Textbox(value=details['instructions'], label="任务指令", interactive=False)
                                    
                                    gr.Textbox(value=details['optimized_prompt'], label="优化后的提示词", interactive=False)
                                    
                                    for key, value in details.items():
                                        if key not in ['signature', 'evaluation_score', 'input_fields', 'output_fields', 'dspy_module', 'llm_model', 'teacher_model', 'optimizer', 'instructions', 'optimized_prompt', 'human_readable_id']:
                                            if isinstance(value, list):
                                                gr.Dropdown(choices=value, value=value, label=key.replace('_', ' ').title(), interactive=False, multiselect=True)
                                            elif isinstance(value, bool):
                                                gr.Checkbox(value=value, label=key.replace('_', ' ').title(), interactive=False)
                                            elif isinstance(value, (int, float)):
                                                gr.Number(value=value, label=key.replace('_', ' ').title(), interactive=False)
                                            else:
                                                gr.Textbox(value=str(value), label=key.replace('_', ' ').title(), interactive=False)
                        

            gr.Markdown("# 查看提示词")
            
            # Add filter and sort functionality in one line
            with gr.Row():
                filter_signature = gr.Dropdown(label="按签名过滤", choices=["全部 (All)"] + unique_signatures, value="全部 (All)", scale=2)
                sort_by = gr.Radio(["运行日期", "评估分数"], label="排序依据", value="运行日期", scale=1)
                sort_order = gr.Radio(["降序", "升序"], label="排序顺序", value="降序", scale=1)

            @gr.render(inputs=[filter_signature, sort_by, sort_order])
            def render_prompts(filter_signature, sort_by, sort_order):
                if filter_signature and filter_signature != "全部 (All)":
                    filtered_prompts = list_prompts(signature_filter=filter_signature)
                else:
                    filtered_prompts = prompts
                
                if sort_by == "评估分数":
                    key_func = lambda x: float(x["Eval Score"])
                else:  # Run Date
                    key_func = lambda x: x["ID"]  # Use the entire ID for sorting
                
                sorted_prompts = sorted(filtered_prompts, key=key_func, reverse=(sort_order == "降序"))
                
                prompt_components = []
                
                for i in range(0, len(sorted_prompts), 3):
                    with gr.Row():
                        for j in range(3):
                            if i + j < len(sorted_prompts):
                                prompt = sorted_prompts[i + j]
                                with gr.Column():
                                    with gr.Group(elem_classes="prompt-card"):
                                        with gr.Column(elem_classes="prompt-details"):
                                            gr.Markdown(f"**ID:** {prompt['ID']}")
                                            gr.Markdown(f"**签名:** {prompt['Signature']}")
                                            gr.Markdown(f"**评估分:** {prompt['Eval Score']}")
                                        view_details_btn = gr.Button("查看详情", elem_classes="view-details-btn", size="sm")
                                    
                                    prompt_components.append((prompt, view_details_btn))
                
                for prompt, btn in prompt_components:
                    btn.click(
                        lambda p=prompt: (p, gr.update(visible=True)),
                        outputs=[selected_prompt, close_details_btn]
                    )

if __name__ == "__main__":
    demo.launch()