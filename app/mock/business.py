import random


def mock_order_snapshot(order_id:str)->dict:
    rng=random.Random(order_id)
    order_status,logistics_status,location=rng.choice([
        ('待发货','暂无物流',None),('运输中','已揽收','上海分拨中心'),
        ('运输中','运输中','杭州转运中心'),('运输中','派送中','本地配送站'),('已完成','已签收','收货地址')])
    amount=round(rng.uniform(29,399),2)
    product_name,category=rng.choice([('喵喵猫粮','宠物食品'),('智能饮水机','宠物用品'),('猫抓板','宠物用品')])
    return {'status':order_status,'amount':amount,'logistics_status':logistics_status,'location':location,
            'product_name':product_name,'category':category,'days_since_receipt':rng.choice([2,5,15]) if order_status=='已完成' else None}


def logistics_result(order_id:str)->dict:
    data=mock_order_snapshot(order_id)
    states=['已揽收','运输中','派送中','已签收']
    tracks=[] if data['logistics_status']=='暂无物流' else [
        {'sequence':i+1,'status':status,'location':data['location'] if status==data['logistics_status'] else ['上海分拨中心','杭州转运中心','本地配送站','收货地址'][i]}
        for i,status in enumerate(states[:states.index(data['logistics_status'])+1])]
    return {'mock':True,'order_id':order_id,'status':data['logistics_status'],'location':data['location'],'tracks':tracks}


def warranty_result(order_id:str)->dict:
    data=mock_order_snapshot(order_id)
    return {'mock':True,'order_id':order_id,'product_name':data['product_name'],
            'status':'在保' if data['days_since_receipt'] is not None else '待签收后核实',
            'basis':'演示规则：签收后365天内在保','days_since_receipt':data['days_since_receipt']}


def return_result(order_id:str)->dict:
    data=mock_order_snapshot(order_id)
    if data['status']!='已完成': return {'mock':True,'order_id':order_id,'found':False}
    return {'mock':True,'order_id':order_id,'status':random.Random('return:'+order_id).choice(['申请待审核','退件运输中','退件待验收']),
            'message':'演示退货记录，不代表已实际申请退款'}
