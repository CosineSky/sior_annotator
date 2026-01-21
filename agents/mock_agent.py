import random

class MockLLMAgent:
    """
    Mock Agent：
    - Judge if a given mask is valid.
    - Give a semantic type.
    """

    SEMANTIC_CANDIDATES = [
        "airport", "ship", "building",
        "road", "farmland", "forest",
        "water", "background"
    ]

    def judge(self, image, mask):
        decision = random.choices(
            ["keep", "discard"],
            weights=[0.75, 0.25]
        )[0]

        semantic = random.choice(self.SEMANTIC_CANDIDATES)

        return {
            "decision": decision,
            "semantic": semantic,
            "confidence": round(random.uniform(0.6, 0.95), 3),
            "reason": f"Mock decision as {semantic}"
        }
