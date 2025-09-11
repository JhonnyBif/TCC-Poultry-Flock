import { FormGroup, FormControl, Validators } from "@angular/forms";

export class MedidasForm extends FormGroup {
  constructor() {
    super({
      superior: new FormControl(null, [Validators.required,Validators.pattern(/^\d*\.?\d+$/)]),
      esquerda: new FormControl(null, [Validators.required,Validators.pattern(/^\d*\.?\d+$/)]),
      baixo: new FormControl(null, [Validators.required,Validators.pattern(/^\d*\.?\d+$/)]),
      direita: new FormControl(null, [Validators.required,Validators.pattern(/^\d*\.?\d+$/)]),
      dataEntrada: new FormControl(null, [Validators.required]),
      dataVideo: new FormControl(null, [Validators.required]),
      idade: new FormControl(null, [Validators.required]),
    });
  }
}
