from strands import tool
import json

# Load knowledge base
with open('data/knowledge_base.json', 'r') as f:
    KB = json.load(f)

@tool
def get_revenue_summary():
    """Get overall revenue metrics including total revenue, opportunity, and model accuracy"""
    return KB['overview']

@tool
def get_material_info(material_id: str):
    """Get detailed information about a specific material by its ID"""
    for mat in KB['material_aggregates']:
        if mat['material_id'] == material_id:
            return mat
    return {"error": f"Material {material_id} not found"}

@tool
def get_top_opportunities(limit: int = 5):
    """Get top revenue opportunities. Default limit is 5."""
    return KB['top_opportunities'][:limit]

@tool
def get_pricing_status():
    """Get pricing status breakdown (underpriced, optimal, overpriced)"""
    return KB['pricing_status']

@tool
def get_margin_analysis():
    """Get high and low margin materials analysis"""
    return KB['margin_analysis']

@tool
def get_material_groups():
    """Get all material groups with their metrics"""
    return KB['material_groups']

@tool
def search_materials(keyword: str):
    """Search materials by name or ID containing the keyword"""
    results = []
    keyword_lower = keyword.lower()
    for mat in KB['material_aggregates']:
        if keyword_lower in mat['material_id'].lower() or keyword_lower in mat['material_name'].lower():
            results.append(mat)
    return results if results else {"error": f"No materials found matching '{keyword}'"}

@tool
def get_underpriced_materials():
    """Get list of underpriced materials that need price increase"""
    underpriced_ids = KB['pricing_status']['underpriced']['materials']
    materials = []
    for mat_id in underpriced_ids[:10]:  # Limit to 10
        for mat in KB['material_aggregates']:
            if mat['material_id'] == mat_id:
                materials.append(mat)
                break
    return materials

@tool
def get_price_ranges():
    """Get min, max, average, and median price information"""
    return KB['price_ranges']

@tool
def get_quantity_analysis():
    """Get order quantity statistics"""
    return KB['quantity_analysis']
