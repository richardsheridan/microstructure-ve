def parse_header(header_row):
    header_sections = header_row.split(',')
    section_name = header_sections[0].strip()
    section_dict = {'section_name':section_name}
    for i in range(1,len(header_sections)):
        section = header_sections[i]
        if '=' in section:
            attr,val = section.split('=')
            section_dict[attr.strip()] = val.strip()
    return section_dict

def save_section(section_dict,count,summary):
    # only save *Element, *Elset, *Solid Section, *Material, *Elastic
    if section_dict['section_name'] == '*Element':
        summary['*Element'] = count
    if section_dict['section_name'] == '*Elset':
        elset_id = section_dict['elset']
        summary[elset_id] =  count
    if section_dict['section_name'] == '*Solid Section':
        elset_id = section_dict['elset']
        mat_id = section_dict['material']
        summary[mat_id] = {'count':summary[elset_id]}
    if section_dict['section_name'] == '*Material':
        mat_id = section_dict['name']
        summary['current_material'] = mat_id
    if section_dict['section_name'] == '*Elastic':
        summary[summary['current_material']]['poisson'] = section_dict['poisson']
    return

def poisson_mixture(summary):
    count,poi_multi_count = 0,0
    for key in summary:
        val = summary[key]
        if isinstance(val,dict) and 'count' in val and 'poisson' in val:
            count += val['count']
            poi_multi_count += val['poisson']*val['count']
    return poi_multi_count/count

def get_poisson(job_name):
    inp_file = job_name + '.inp'
    inp = open(inp_file,'r')
    prev_section = None
    elset = {}
    summary = {}
    for row in inp:
        if row.startswith('*') and not row.startswith('**'):
            section_dict = parse_header(row) # parse section header
            # if prev_section exists, save it with the row count
            if prev_section:
                save_section(prev_section,count,summary)
            count = 0 # reset row count
            prev_section = section_dict # save current section_dict
        else:
            count += 1
            # *Elastic then save the poisson's ratio
            if prev_section['section_name'] == '*Elastic':
                prev_section['poisson'] = float(row.split(',')[1].strip())
    inp.close()
    return poisson_mixture(summary)

if __name__ == '__main__':
    print(get_poisson('example'))
