"""Vetted guidance prevents complete computed answers leaking through hint modes."""
import json

from app.schemas.math_solver import MathSolverResult
from app.rag.prompt_builder import ChatMode
from app.tools.solver_router import fold


def guidance_steps(result: MathSolverResult, query: str = "") -> list[str]:
    if any(word in fold(query) for word in ("giai", "tinh", "rut gon", "dao ham", "nguyen ham")):
        vietnamese_steps = {
            "solve_equation": [
                "Chuyển các hạng tử về một vế rồi thu gọn các hạng tử đồng dạng.",
                "Thử phân tích biểu thức thành nhân tử; nếu là phương trình bậc hai, em cũng có thể dùng công thức nghiệm.",
                "Cho từng nhân tử bằng không rồi kiểm tra nghiệm trong phương trình ban đầu. Em hãy tự hoàn thành bước cuối này.",
            ],
            "differentiate": [
                "Lấy đạo hàm từng hạng tử bằng quy tắc đạo hàm của tổng.",
                "Với lũy thừa của biến, nhân với số mũ rồi giảm số mũ đi một; đạo hàm của hằng số bằng không.",
                "Em hãy tự ghép các đạo hàm vừa tìm được để hoàn thành kết quả.",
            ],
            "integrate": [
                "Tìm nguyên hàm của từng hạng tử riêng biệt.",
                "Với lũy thừa có số mũ khác âm một, tăng số mũ lên một rồi chia cho số mũ mới.",
                "Ghép các hạng tử và thêm hằng số tùy ý. Lấy đạo hàm biểu thức của em để kiểm tra.",
            ],
            "simplify": [
                "Tìm các nhân tử chung trước khi rút gọn.",
                "Phân tích tử và mẫu thành nhân tử, đồng thời giữ điều kiện mẫu ban đầu khác không.",
                "Rút gọn nhân tử chung khác không rồi tự viết biểu thức cuối cùng.",
            ],
            "calculate": [
                "Thực hiện đúng thứ tự phép tính, bắt đầu từ ngoặc và lũy thừa.",
                "Tính các hàm như căn bậc hai trước khi kết hợp những hạng tử còn lại.",
                "Em hãy tự hoàn thành phép tính còn lại và kiểm tra từng bước.",
            ],
        }
        return vietnamese_steps[result.problem_type]
    steps = {
        "solve_equation": [
            "Move all terms to one side and collect like terms.",
            "Try factoring the expression; if it is quadratic, you can also use the quadratic formula.",
            "Set each factor equal to zero, then check your candidates in the original equation. Complete this last step yourself.",
        ],
        "differentiate": [
            "Differentiate one term at a time using the sum rule.",
            "For a power of the variable, multiply by its exponent and reduce the exponent by one; constants have derivative zero.",
            "Combine the differentiated terms yourself to finish.",
        ],
        "integrate": [
            "Integrate each term separately.",
            "For a power other than minus one, increase the exponent by one and divide by that new exponent.",
            "Combine your terms and include an arbitrary constant. Differentiate your expression to check it.",
        ],
        "simplify": [
            "Look for common factors before cancelling anything.",
            "Factor the numerator and denominator, keeping the original denominator's excluded values.",
            "Cancel common nonzero factors and finish the reduced expression yourself.",
        ],
        "calculate": [
            "Follow the order of operations, starting with parentheses and powers.",
            "Evaluate functions such as square roots before combining the remaining terms.",
            "Complete the remaining arithmetic yourself and check each operation.",
        ],
    }
    return steps[result.problem_type]


async def generate_guidance(provider, assembled, result, chat_mode: ChatMode, query: str = "") -> str:
    """The LLM chooses vetted next steps; arbitrary generated prose never leaks roots.

    It sees the complete internal result but returns only indices. Invalid output
    or provider failure uses deterministic guidance, keeping chat available.
    """
    steps = guidance_steps(result, query)
    count = 1 if chat_mode == ChatMode.LIGHT_GUIDANCE else len(steps)
    indices = list(range(count))
    try:
        selection = await provider.generate(
            prompt=assembled.prompt + "\n<VETTED_HINTS>\n" + json.dumps(steps) +
                   '\n</VETTED_HINTS>\nReturn only JSON {"hint_indices": [0, ...]}. Select conceptual next steps; never output a final answer.',
            system_instruction=assembled.system_instruction,
        )
        parsed = json.loads(selection)
        selected = parsed["hint_indices"]
        if (isinstance(selected, list) and len(selected) == count
                and all(type(index) is int and 0 <= index < len(steps) for index in selected)
                and len(set(selected)) == count):
            indices = selected
    except Exception:
        pass
    text = "\n\n".join(steps[index] for index in sorted(indices))
    if assembled.sources:
        text += " [Source 1]"
    return text


def verified_final(result: MathSolverResult) -> str:
    if isinstance(result.result, list):
        text = f"{result.variable} ∈ {{{', '.join(result.result)}}}" if result.result else "No real solutions."
    else:
        text = str(result.result)
        if result.problem_type == "integrate":
            text += " + C"
    return "Verified result: " + text
