import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class ProductMemory:
    """Holds information about a product that has been viewed."""
    product_name: str
    url: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    summary: str = ""

@dataclass
class MemoryModule:
    """A module to remember which products have been seen and their analysis."""
    search_queries: List[str] = field(default_factory=list)
    products: List[ProductMemory] = field(default_factory=list)

    def add_product(self, product: ProductMemory):
        """Adds a product to the memory, avoiding duplicates."""
        if not self.get_product_by_url(product.url):
            self.products.append(product)

    def add_search_query(self, query: str):
        """Adds a unique search query to memory."""
        q = query.strip()
        if q and q not in self.search_queries:
            self.search_queries.append(q)

    def is_product_in_memory(self, product_name: str) -> bool:
        """Checks if a product is already in memory by its name."""
        return any(p.product_name.lower() in product_name.lower() or product_name.lower() in p.product_name.lower() for p in self.products)

    def get_memory_summary_for_prompt(self) -> str:
        """Returns a string summary of the products in memory."""
        if not self.products:
            return "No products have been viewed yet."
        
        summary_lines = []
        for product in self.products:
            summary_lines.append(f"- {product.product_name}")
            # if product.pros:
            #     summary_lines.append(f"  Pros: {', '.join(product.pros)}")
            # if product.cons:
            #     summary_lines.append(f"  Cons: {', '.join(product.cons)}")
            # if product.summary:
            #     summary_lines.append(f"  Summary: {product.summary}")
        return "\n".join(summary_lines)

    def get_search_history_summary(self) -> str:
        """Returns searched queries as a newline-separated string."""
        if not self.search_queries:
            return "No searches performed yet."
        return "\n".join([f"- {q}" for q in self.search_queries])

    def get_product_by_url(self, url: str) -> Optional[ProductMemory]:
        """Gets a product from memory by its URL."""
        for product in self.products:
            if product.url == url:
                return product
        return None

    def save_to_json(self, file_path: str):
        """Saves the memory to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=2) 