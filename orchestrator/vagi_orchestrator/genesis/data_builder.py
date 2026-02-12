from __future__ import annotations


def build_dialogue_intent_samples(repeats: int = 32) -> list[str]:
    base_pairs = [
        (
            "Xin chao, toi can ban giup kiem tra code dang loi.",
            "[intent:scan_code] Da ro. Gui file hoac stack trace de toi phan tich.",
        ),
        (
            "Hay viet cho toi mot ham python de cong hai so.",
            "[intent:code_generate] def add(a, b): return a + b",
        ),
        (
            "Toi muon toi uu bo nho cho API service.",
            "[intent:optimize_memory] Uu tien stream processing va gioi han buffer tam.",
        ),
        (
            "Can toi uu toc do response cua endpoint chat.",
            "[intent:optimize_speed] Giam allocation, cache ket qua va bat timeout guard.",
        ),
        (
            "Ban co the giai thich loi TypeError trong python khong?",
            "[intent:explain_error] TypeError xay ra khi phep toan dung sai kieu du lieu.",
        ),
        (
            "Please help me run tests safely before release.",
            "[intent:run_tests] Start with unit tests, then integration tests, then smoke checks.",
        ),
        (
            "Write a polite response when user asks for progress update.",
            "[intent:user_communication] Trang thai hien tai da duoc cap nhat ngan gon va ro rang.",
        ),
        (
            "How to secure login flow?",
            "[intent:secure_design] Validate input, hash password, rate limit, audit every login attempt.",
        ),
        (
            "Toi can sua bug nhung khong duoc pha vo API cu.",
            "[intent:backward_compat] Giu nguyen contract va bo sung test hoi quy bat buoc.",
        ),
        (
            "Plan deployment with rollback.",
            "[intent:release_plan] Trien khai theo canary, theo doi metric, rollback neu regression tang.",
        ),
        (
            "Tao cho toi huong dan dung verifier an toan.",
            "[intent:verifier_usage] Luon chay verifier truoc khi act va chan output nguy hiem.",
        ),
        (
            "Can mot cau tra loi lich su va gon gang.",
            "[intent:conversation] Da ro. Minh se tra loi ro rang, ngan gon va dung trong tam.",
        ),
    ]
    samples: list[str] = []
    for _ in range(repeats):
        for user, assistant in base_pairs:
            samples.append(f"User: {user}\nAssistant: {assistant}")
    return samples

