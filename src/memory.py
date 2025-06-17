from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ProductMemory:
    """Holds information about a product that has been viewed."""
    product_name: str
    url: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    summary: str = ""

class MemoryModule:
    """A module to remember which products have been seen and their analysis."""
    def __init__(self):
        self.products: List[ProductMemory] = []

    def add_product(self, product: ProductMemory):
        """Adds a product to the memory."""
        self.products.append(product)

    def is_product_in_memory(self, product_name: str) -> bool:
        """Checks if a product is already in memory by its name."""
        return any(p.product_name.lower() in product_name.lower() or product_name.lower() in p.product_name.lower() for p in self.products)

    def get_memory_summary_for_prompt(self) -> str:
        """Returns a string summary of the products in memory."""
        if not self.products:
            return "No products have been viewed yet."
        
        summary_lines = []
        for idx, product in enumerate(self.products):
            summary_lines.append(f"{idx+1}. {product.product_name}")
            # if product.pros:
            #     summary_lines.append(f"  Pros: {', '.join(product.pros)}")
            # if product.cons:
            #     summary_lines.append(f"  Cons: {', '.join(product.cons)}")
            # if product.summary:
            #     summary_lines.append(f"  Summary: {product.summary}")
        return "\n".join(summary_lines)

    def get_product_by_url(self, url: str) -> Optional[ProductMemory]:
        """Gets a product from memory by its URL."""
        for product in self.products:
            if product.url == url:
                return product
        return None 