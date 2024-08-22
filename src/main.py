import click
import HAR_expt as hr
import mimic_expt as mm

@click.command()
@click.option("-d", "--data", required=True, type=str,
              help="HAR/MIMIC", default='HAR')
@click.option("-l", "--model_type", required=True, type=str,
              help="local/global", default='local')
def main(data, model_type):
    if data=='HAR' and model_type=='local':
        print("Dataset selected HAR, generating local explanations")
        hr.local_explain_HAR()
        hr.evaluate_explainer(path='all_masks_local_HAR.pkl')

    elif data=='HAR' and model_type=='global':
        print("Dataset selected HAR, generating global explanations")
        hr.global_explain_HAR()
        hr.evaluate_explainer(path='all_masks_global_HAR.pkl')

    elif data=='MIMIC' and model_type=='local':
        print("Dataset selected MIMIC, generating local explanations")
        mm.local_explain_MIMIC()
        mm.evaluate_explainer(path='all_masks_local_MIMIC.pkl')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()